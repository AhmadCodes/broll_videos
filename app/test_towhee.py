# %%
import os
import numpy as np



from towhee import ops, pipe, DataCollection
# pip for video embeddings
pipe_video = (
    pipe.input('video_path')
    .map('video_path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 12}))
    .map('frames', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='video', device='cuda'))
    # .map('vec', 'vec', ops.normalize)
    .output('vec')
)
# %%
pipe_video("/workspace/broll_videos/app/static/test.mp4").get()

# %%

pipe_text = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cuda'))
    .output('vec')
)
# %%

