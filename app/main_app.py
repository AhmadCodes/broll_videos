# %%

# from moviepy.config import change_settings
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

from pexelsapi.pexels import Pexels
import uuid
from io import BytesIO
import boto3
from dotenv import load_dotenv
import requests
import json
from towhee import ops, pipe
import scipy as sc

import os

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), "..", '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_s3_client():
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return s3_client


# Set your Pexels API key
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Create an API object
pexel_api = Pexels(PEXELS_API_KEY)

# Search for videos with the keyword 'ocean'


# %%

video_embeddings_pipe = (
    pipe.input('video_path')
    .map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
    .map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda'))
    # .map('vec', 'vec', ops.normalize)
    .output('vec')
)

text_embeddings_pipe = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda'))
    .output('vec')
)

S3_CLIENT = get_s3_client()
# %%


def convert_to_text(word_level_transcript):
    text = " ".join([w['word'] for w in word_level_transcript])
    return text


def give_context(word_level_transcript,
                 context_start_s,
                 context_end_s,
                 context_buffer_s=5):
    context = ""
    for word in word_level_transcript:
        if float(word['start']) >= context_start_s - context_buffer_s and float(word['end']) <= context_end_s + context_buffer_s:
            context += word['word'] + " "
    return context


# %%
chatgpt_url = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%


def download_pexels_video(search_phrase,
                          download_dir = os.path.join(os.path.dirname(__file__), "static")):
    search_videos = pexel_api.search_videos(query=search_phrase, orientation='', size='', color='', locale='', page=1, per_page=5)
    print(search_videos)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    downloaded_files = []
    for video in search_videos["videos"]:
        video_id = video["id"]
        download_url = f'https://www.pexels.com/video/{video_id}/download'
        response = requests.get(download_url)
        
        if response.status_code == 200:
            file_name = os.path.join(download_dir, f'{video_id}.mp4')
            with open(file_name, 'wb') as file:
                file.write(response.content)
            downloaded_files.append(file_name)
            print(f'Downloaded video: {file_name}')
        else:
            print(f'Failed to download video: {video_id}')
            
    return downloaded_files


def get_video_embeddings(video_path):
    return video_embeddings_pipe(video_path).get()


def get_text_embeddings(sentence):
    return text_embeddings_pipe(sentence).get()


def rank_videos(video_paths,
                sentence,
                top_K=5):
    sentence_embedding = get_text_embeddings(sentence)

    video_embeddings = []
    for video_path in video_paths:
        video_embeddings.append(get_video_embeddings(video_path))

    video_embeddings = sc.array(video_embeddings)
    sentence_embedding = sc.array([sentence_embedding])

    # Compute the similarity between the sentence and each video using L2 distance
    similarity = sc.linalg.norm(video_embeddings - sentence_embedding, axis=1)

    # Rank the video paths based on the similarity
    ranked_video_paths = [video_paths[i] for i in similarity.argsort()]

    return ranked_video_paths[:top_K]


# %%


def validate_KV_pair(dict_list,
                     debug=False):
    for d in dict_list:
        check_all_keys = all([k in d.keys() for k in ['description']])

        check_description = isinstance(d['description'], str)

        if debug:
            print("check_all_keys: ", check_all_keys)
            print("check_description: ", check_description)

        return check_all_keys and check_description


def json_corrector(json_str,
                   exception_str,
                   openaiapi_key):

    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}

    prompt_prefix = f"""Exception: {exception_str}
    JSON:{json_str}
    ------------------
    """
    prompt = prompt_prefix + """\n Correct the following JSON, eliminate any formatting issues occured due to misplaces or lack or commas, brackets, semicolons, colons, other symbols, etc
    \nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert in correcting JSON strings, you return a VALID JSON by eliminating all formatting issues"},
        {"role": "user", "content": prompt}
    ]
    chatgpt_payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }

    try:
        url = chatgpt_url
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ",
                  response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)

            return None
        # Extract data from the API's response
        try:
            output = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            return output
        except Exception as e:
            print("Error in response from OPENAI GPT-3.5: ", e)
            return None

    except Exception as e:
        return None


def fetch_broll_description(wordlevel_info,
                            context_start_s,
                            context_end_s,
                            context_buffer_s,
                            url,
                            openaiapi_key,
                            debug=False):

    success = False
    err_msg = ""

    if openaiapi_key == "":
        openaiapi_key = OPENAI_API_KEY

    assert openaiapi_key != "", "Please enter your OPENAI API KEY"

    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}

    transcript = convert_to_text(wordlevel_info)
    context = give_context(wordlevel_info, context_start_s,
                           context_end_s, context_buffer_s)

    prompt_prefix = """Complete Transcript:
    {}
    ----
    Time Stamped Context:
    {}
    ----
    
    Given the Transcript of a video, generate very relevant stock image description to insert as B-roll image.
    The description of B-roll images should perfectly match with the context window that is provided.
    Strictly don't include any exact word or text labels to be depicted in the image.
    Strictly output only JSON in the output using the format (BE CAREFUL NOT TO MISS ANY COMMAS, QUOTES OR SEMICOLONS ETC)-""".format(transcript,
                                                                                                                                      context)

    sample = {"description": "..."}

    prompt = prompt_prefix + json.dumps(sample) + f"""\n
    Be sure to only make 1 json. \nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts."},
        {"role": "user", "content": prompt}
    ]

    chatgpt_payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }
    while not success:
        success = False
        # Make the request to OpenAI's API
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ",
                  response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)
            if 'error' in response_json:
                err_msg = response_json['error']
                return None, err_msg
            success = False
            continue
        # Extract data from the API's response
        try:
            output_ = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            output = [output_]
            if debug:
                print("output: ", output)
            success = validate_KV_pair(output, debug=debug)
            if success:
                print("JSON: ", output)
                success = True
            else:
                print("Could not validate Key-Value pairs in JSON")
                print("Trying again...")
                success = False
                continue
        except Exception as e:
            print("Error in response from OPENAI GPT-4: ", e)

            output = json_corrector(response_json['choices'][0]['message']['content'].strip(),
                                    str(e),
                                    openaiapi_key)
            if output is not None:
                print("Corrected JSON: ", output)
                success = True
            else:
                print("Could not correct JSON")
                print("Trying again...")
                success = False
                continue

    return output, err_msg

# %%


# %%


def upload_image_to_s3(image, bucket, s3_client):
    # Convert PIL Image to Bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_byte_array = buffer.getvalue()
    # uuid for uploaded_image as the image name
    upload_key = uuid.uuid4().hex + ".png"
    # Upload Bytes to S3
    s3_client.put_object(Body=image_byte_array, Bucket=bucket, Key=upload_key)

    # Generate the URL of the uploaded image
    s3_img_info = {
        "bucket": bucket,
        "key": upload_key
    }

    return s3_img_info

# %%


def pipeline(word_level_transcript,
             context_start_s=0,
             context_end_s=0,
             context_buffer_s=5,
             broll_image_steps=50,
             openaiapi_key=os.getenv("OPENAI_API_KEY"),
             debug=False
             ):

    # Fetch B-roll descriptions
    broll_descriptions, err_msg = fetch_broll_description(word_level_transcript,
                                                          context_start_s,
                                                          context_end_s,
                                                          context_buffer_s,
                                                          chatgpt_url,
                                                          openaiapi_key,
                                                          debug=debug)
    if debug:
        print("B-roll descriptions: ", broll_descriptions)
    if err_msg != "" and broll_descriptions is None:
        return err_msg

    # Generate B-roll images

    img_upload_info = []
    for i, img in enumerate(allimages):
        img_info = upload_image_to_s3(img, "brollimages", S3_CLIENT)
        img_info['description'] = broll_descriptions[i]['description']
        img_upload_info.append(img_info)

    return img_upload_info
# %%


if __name__ == "__main__":
    from example import example_transcript

    context_start_s = 12
    context_end_s = 30
    context_buffer_s = 5

    img_info = pipeline(example_transcript,
                        context_start_s=context_start_s,
                        context_end_s=context_end_s,
                        context_buffer_s=context_buffer_s,
                        broll_image_steps=50,
                        openaiapi_key=OPENAI_API_KEY,
                        debug=True)
# %%
