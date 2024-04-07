'''
Contains the handler function that will be called by the serverless.
'''
#%%
import os
import base64
import concurrent.futures

from main_app import pipeline

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA
import torch
torch.cuda.empty_cache()


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    word_level_transcript = job_input['word_level_transcript']
    context_start_s = job_input.get('context_start_s', 0)
    context_end_s = job_input.get('context_end_s', 0)
    context_buffer_s = job_input.get('context_buffer_s', 0)
    n_search_phrases = job_input.get('n_search_phrases', 3)
    n_searche_results_per_phrase = job_input.get('n_searche_results_per_phrase', 5)
    top_K = job_input.get('top_K', 5)
    
    if context_start_s < 0 or context_end_s < 0 or context_start_s > context_end_s:
        return {"error": "Invalid context_start_s or context_end_s"}
    
    results = pipeline(word_level_transcript=word_level_transcript,
                       context_start_s=context_start_s,
                       context_end_s=context_end_s,
                          context_buffer_s=context_buffer_s,
                          n_seach_phrases=n_search_phrases,
                          n_searches_per_phrase=n_searche_results_per_phrase,top_K=top_K,
                          debug=True)
    

    return results
#%%
# 
job = {
    "input": {
        "word_level_transcript": [{"end":0.92,"start":0,"word":" Hey,"},{"end":1.3,"start":0.92,"word":" curious"},{"end":1.84,"start":1.3,"word":" kids."},{"end":2.42,"start":2.42,"word":" Kevin"},{"end":2.6,"start":2.42,"word":" J."},{"end":2.86,"start":2.6,"word":" DeBroon"},{"end":3.14,"start":2.86,"word":" here."},{"end":3.4,"start":3.34,"word":" I"},{"end":3.5,"start":3.4,"word":" am"},{"end":3.62,"start":3.5,"word":" a"},{"end":3.98,"start":3.62,"word":" former"},{"end":4.5,"start":3.98,"word":" NASA"},{"end":4.84,"start":4.5,"word":" rocket"},{"end":5.32,"start":4.84,"word":" scientist"},{"end":5.78,"start":5.32,"word":" and"},{"end":6.14,"start":5.78,"word":" today"},{"end":6.7,"start":6.14,"word":" I"},{"end":6.86,"start":6.7,"word":" am"},{"end":7.02,"start":6.86,"word":" going"},{"end":7.16,"start":7.02,"word":" to"},{"end":7.5,"start":7.16,"word":" answer"},{"end":7.7,"start":7.5,"word":" my"},{"end":8.12,"start":7.7,"word":" most"},{"end":8.7,"start":8.12,"word":" asked"},{"end":9.5,"start":8.7,"word":" question."},{"end":10.18,"start":9.84,"word":" How"},{"end":10.38,"start":10.18,"word":" many"},{"end":11.26,"start":10.38,"word":" Earths"},{"end":11.38,"start":11.26,"word":" could"},{"end":11.8,"start":11.38,"word":" fit"},{"end":12.26,"start":11.8,"word":" inside"},{"end":12.5,"start":12.26,"word":" of"},{"end":12.74,"start":12.5,"word":" this"},{"end":13.06,"start":12.74,"word":" Sun?"},{"end":18.88,"start":18.360000000000003,"word":" Let's"},{"end":19.04,"start":18.88,"word":" have"},{"end":19.22,"start":19.04,"word":" you"},{"end":19.36,"start":19.22,"word":" guess"},{"end":19.8,"start":19.36,"word":" first."},{"end":20.16,"start":19.96,"word":" What"},{"end":20.22,"start":20.16,"word":" do"},{"end":20.28,"start":20.22,"word":" you"},{"end":20.54,"start":20.28,"word":" think?"},{"end":21.2,"start":20.98,"word":" How"},{"end":21.46,"start":21.2,"word":" many?"},{"end":22.12,"start":21.88,"word":" You"},{"end":22.24,"start":22.12,"word":" have"},{"end":22.4,"start":22.24,"word":" your"},{"end":22.64,"start":22.4,"word":" guess?"},{"end":23.12,"start":23.02,"word":" Write"},{"end":23.26,"start":23.12,"word":" it"},{"end":23.56,"start":23.26,"word":" down"},{"end":23.88,"start":23.56,"word":" so"},{"end":23.96,"start":23.88,"word":" you"},{"end":24.12,"start":23.96,"word":" can"},{"end":24.22,"start":24.12,"word":" go"},{"end":24.46,"start":24.22,"word":" back"},{"end":24.62,"start":24.46,"word":" and"},{"end":24.82,"start":24.62,"word":" check"},{"end":25.04,"start":24.82,"word":" later"},{"end":25.18,"start":25.04,"word":" to"},{"end":25.38,"start":25.18,"word":" see"},{"end":25.54,"start":25.38,"word":" how"},{"end":25.9,"start":25.54,"word":" close"},{"end":26.24,"start":25.9,"word":" or"},{"end":26.64,"start":26.24,"word":" how"},{"end":26.98,"start":26.64,"word":" far"},{"end":27.6,"start":26.98,"word":" you"},{"end":28,"start":27.6,"word":" are."},{"end":31.96,"start":31.42,"word":" Now,"},{"end":32.28,"start":32.08,"word":" our"},{"end":32.62,"start":32.28,"word":" solar"},{"end":33.04,"start":32.62,"word":" system"},{"end":33.4,"start":33.04,"word":" is"},{"end":33.66,"start":33.4,"word":" huge"},{"end":34.22,"start":33.66,"word":" and"},{"end":34.48,"start":34.22,"word":" I"},{"end":34.98,"start":34.48,"word":" mean"},{"end":35.64,"start":34.98,"word":" huge."},{"end":36.22,"start":36.18,"word":" It's"},{"end":36.38,"start":36.22,"word":" full"},{"end":36.68,"start":36.38,"word":" of"},{"end":36.86,"start":36.68,"word":" so"},{"end":37.18,"start":36.86,"word":" many"},{"end":37.4,"start":37.18,"word":" cool"},{"end":37.6,"start":37.4,"word":" and"},{"end":37.92,"start":37.6,"word":" unique"},{"end":38.56,"start":37.92,"word":" things."},{"end":39.06,"start":38.78,"word":" Things"},{"end":39.16,"start":39.06,"word":" that"},{"end":39.2,"start":39.16,"word":" are"},{"end":39.54,"start":39.2,"word":" all"},{"end":39.7,"start":39.54,"word":" different"},{"end":40.02,"start":39.7,"word":" shapes"},{"end":40.28,"start":40.02,"word":" and"},{"end":40.56,"start":40.28,"word":" sizes."},{"end":41.34,"start":41.14,"word":" But"},{"end":41.56,"start":41.34,"word":" how"},{"end":41.96,"start":41.56,"word":" different"},{"end":42.2,"start":41.96,"word":" are"},{"end":42.5,"start":42.2,"word":" these"},{"end":42.78,"start":42.5,"word":" sizes"},{"end":43.2,"start":42.78,"word":" really?"},{"end":44,"start":43.82,"word":" Well,"},{"end":44.42,"start":44.12,"word":" if"},{"end":44.68,"start":44.42,"word":" we"},{"end":44.9,"start":44.68,"word":" shrink"},{"end":45.22,"start":44.9,"word":" down"},{"end":45.44,"start":45.22,"word":" the"},{"end":45.7,"start":45.44,"word":" Sun"},{"end":45.88,"start":45.7,"word":" to"},{"end":46.08,"start":45.88,"word":" the"},{"end":46.4,"start":46.08,"word":" size"},{"end":46.68,"start":46.4,"word":" of"},{"end":47.02,"start":46.68,"word":" this"},{"end":47.58,"start":47.02,"word":" inflatable"},{"end":48.22,"start":47.58,"word":" ball,"},{"end":48.8,"start":48.36,"word":" how"},{"end":48.96,"start":48.8,"word":" big"},{"end":49.06,"start":48.96,"word":" would"},{"end":49.26,"start":49.06,"word":" each"},{"end":49.46,"start":49.26,"word":" planet"},{"end":49.7,"start":49.46,"word":" be?"},{"end":50.46,"start":50.18,"word":" Actually,"},{"end":50.72,"start":50.6,"word":" let's"},{"end":50.94,"start":50.72,"word":" not"},{"end":51.22,"start":50.94,"word":" use"},{"end":51.34,"start":51.22,"word":" the"},{"end":51.58,"start":51.34,"word":" word"},{"end":52,"start":51.58,"word":" big."},{"end":52.44,"start":52.3,"word":" How"},{"end":52.94,"start":52.44,"word":" small,"},{"end":53.5,"start":53.06,"word":" tiny,"},{"end":54.08,"start":53.58,"word":" little"},{"end":54.52,"start":54.08,"word":" would"},{"end":54.7,"start":54.52,"word":" the"},{"end":54.86,"start":54.7,"word":" planet"},{"end":55.18,"start":54.86,"word":" speed?"},{"end":56.6,"start":56.06,"word":" This"},{"end":57.36,"start":56.6,"word":" small."},{"end":58.7,"start":58.16,"word":" This"},{"end":58.84,"start":58.7,"word":" is"},{"end":58.88,"start":58.84,"word":" a"},{"end":59.34,"start":58.88,"word":" 4"},{"end":59.58,"start":59.34,"word":" billion"},{"end":59.84,"start":59.58,"word":" to"},{"end":59.98,"start":59.84,"word":" one"},{"end":60.38,"start":59.98,"word":" scale,"},{"end":60.88,"start":60.6,"word":" which"},{"end":61.48,"start":60.88,"word":" means"},{"end":62.16,"start":61.48,"word":" that"},{"end":62.32,"start":62.16,"word":" out"},{"end":62.48,"start":62.32,"word":" in"},{"end":62.88,"start":62.48,"word":" space,"},{"end":63.28,"start":62.88,"word":" these"},{"end":63.44,"start":63.28,"word":" are"},{"end":63.9,"start":63.44,"word":" actually"},{"end":64.8,"start":63.9,"word":" 4"},{"end":65.4,"start":64.8,"word":" billion"},{"end":66.22,"start":65.4,"word":" times"},{"end":66.8,"start":66.22,"word":" larger."},{"end":67.3,"start":67.08,"word":" Let's"},{"end":67.5,"start":67.3,"word":" zoom"},{"end":67.64,"start":67.5,"word":" in"},{"end":67.78,"start":67.64,"word":" and"},{"end":67.94,"start":67.78,"word":" check"},{"end":68.12,"start":67.94,"word":" out"},{"end":68.18,"start":68.12,"word":" how"},{"end":68.32,"start":68.18,"word":" the"},{"end":68.68,"start":68.32,"word":" planets"},{"end":69.06,"start":68.68,"word":" compare"},{"end":69.34,"start":69.06,"word":" to"},{"end":69.5,"start":69.34,"word":" each"},{"end":69.7,"start":69.5,"word":" other"},{"end":69.84,"start":69.7,"word":" at"},{"end":70.02,"start":69.84,"word":" this"},{"end":70.46,"start":70.02,"word":" scale."},{"end":71,"start":70.86,"word":" Like"},{"end":71.18,"start":71,"word":" look"},{"end":71.32,"start":71.18,"word":" at"},{"end":71.56,"start":71.32,"word":" Mercury."},{"end":72.64,"start":72.16,"word":" Smallest"},{"end":72.82,"start":72.64,"word":" planet"},{"end":73,"start":72.82,"word":" we"},{"end":73.22,"start":73,"word":" have."},{"end":73.4,"start":73.24,"word":" Look"},{"end":73.56,"start":73.4,"word":" how"},{"end":73.88,"start":73.56,"word":" tiny"},{"end":74.16,"start":73.88,"word":" that"},{"end":74.3,"start":74.16,"word":" that"},{"end":74.72,"start":74.3,"word":" is."},{"end":75.28,"start":75.06,"word":" And"},{"end":75.32,"start":75.28,"word":" then"},{"end":75.44,"start":75.32,"word":" moving"},{"end":75.66,"start":75.44,"word":" up"},{"end":75.9,"start":75.66,"word":" to"},{"end":76.12,"start":75.9,"word":" Venus"},{"end":76.56,"start":76.12,"word":" and"},{"end":77.28,"start":76.56,"word":" Earth,"},{"end":77.6,"start":77.28,"word":" about"},{"end":77.84,"start":77.6,"word":" the"},{"end":78.2,"start":77.84,"word":" same"},{"end":78.6,"start":78.2,"word":" size"},{"end":78.92,"start":78.6,"word":" there."},{"end":79.16,"start":78.92,"word":" Still"},{"end":79.62,"start":79.16,"word":" very"},{"end":80.08,"start":79.62,"word":" small"},{"end":80.3,"start":80.08,"word":" like"},{"end":80.42,"start":80.3,"word":" the"},{"end":80.64,"start":80.42,"word":" size"},{"end":80.84,"start":80.64,"word":" of"},{"end":80.9,"start":80.84,"word":" the"},{"end":81.18,"start":80.9,"word":" baby."},{"end":82.02,"start":81.72,"word":" Mars"},{"end":82.3,"start":82.02,"word":" is"},{"end":82.72,"start":82.3,"word":" super"},{"end":83.12,"start":82.72,"word":" tiny"},{"end":83.44,"start":83.12,"word":" as"},{"end":83.66,"start":83.44,"word":" well."},{"end":84.12,"start":83.88,"word":" Then"},{"end":84.3,"start":84.12,"word":" moving"},{"end":84.48,"start":84.3,"word":" to"},{"end":84.66,"start":84.48,"word":" our"},{"end":84.98,"start":84.66,"word":" biggest"},{"end":85.38,"start":84.98,"word":" planet"},{"end":85.76,"start":85.38,"word":" Jupiter."},{"end":86.52,"start":86.24,"word":" But"},{"end":86.72,"start":86.52,"word":" wait,"},{"end":86.86,"start":86.86,"word":" hold"},{"end":87,"start":86.86,"word":" up."},{"end":87.18,"start":87.1,"word":" You're"},{"end":87.38,"start":87.18,"word":" telling"},{"end":87.56,"start":87.38,"word":" me"},{"end":87.82,"start":87.56,"word":" that"},{"end":88.06,"start":87.82,"word":" Jupiter"},{"end":88.62,"start":88.06,"word":" has"},{"end":88.98,"start":88.62,"word":" rings"},{"end":89.38,"start":88.98,"word":" just"},{"end":89.66,"start":89.38,"word":" like"},{"end":89.94,"start":89.66,"word":" Saturn."},{"end":90.52,"start":90.42,"word":" Yeah."},{"end":91.24,"start":90.96,"word":" And"},{"end":91.58,"start":91.24,"word":" Uranus"},{"end":91.88,"start":91.58,"word":" and"},{"end":92.2,"start":91.88,"word":" Neptune."},{"end":93.94,"start":93.4,"word":" Saturn's"},{"end":94.06,"start":93.94,"word":" are"},{"end":94.48,"start":94.06,"word":" just"},{"end":94.64,"start":94.48,"word":" the"},{"end":95.12,"start":94.64,"word":" biggest."},{"end":95.8,"start":95.46,"word":" Move"},{"end":95.92,"start":95.8,"word":" on"},{"end":95.98,"start":95.92,"word":" to"},{"end":96.28,"start":95.98,"word":" Saturn."},{"end":96.54,"start":96.4,"word":" It's"},{"end":96.54,"start":96.54,"word":" a"},{"end":96.8,"start":96.54,"word":" little"},{"end":97,"start":96.8,"word":" bit"},{"end":97.34,"start":97,"word":" smaller"},{"end":97.7,"start":97.34,"word":" than"},{"end":97.96,"start":97.7,"word":" Jupiter."},{"end":98.44,"start":98.36,"word":" And"},{"end":98.54,"start":98.44,"word":" then"},{"end":98.9,"start":98.54,"word":" Uranus"},{"end":99.02,"start":98.9,"word":" and"},{"end":99.32,"start":99.02,"word":" Neptune"},{"end":99.48,"start":99.32,"word":" are"},{"end":99.72,"start":99.48,"word":" both"},{"end":99.9,"start":99.72,"word":" the"},{"end":100.22,"start":99.9,"word":" same"},{"end":100.6,"start":100.22,"word":" size."},{"end":101.16,"start":100.84,"word":" But"},{"end":101.48,"start":101.16,"word":" this"},{"end":101.68,"start":101.48,"word":" is"},{"end":101.84,"start":101.68,"word":" how"},{"end":102,"start":101.84,"word":" they"},{"end":102.38,"start":102,"word":" compare"},{"end":102.62,"start":102.38,"word":" to"},{"end":102.8,"start":102.62,"word":" each"},{"end":103.04,"start":102.8,"word":" other"},{"end":103.26,"start":103.04,"word":" if"},{"end":103.42,"start":103.26,"word":" they"},{"end":103.74,"start":103.42,"word":" were"},{"end":104.7,"start":103.74,"word":" 4"},{"end":105.56,"start":104.7,"word":" billion"},{"end":106.38,"start":105.56,"word":" times"},{"end":107,"start":106.38,"word":" smaller."},{"end":108.24,"start":107,"word":" Wow."},{"end":108.88,"start":108.6,"word":" Like"},{"end":109.04,"start":108.88,"word":" I"},{"end":109.34,"start":109.04,"word":" said,"},{"end":109.78,"start":109.58,"word":" our"},{"end":110,"start":109.78,"word":" solar"},{"end":110.46,"start":110,"word":" system"},{"end":111.06,"start":110.46,"word":" is"},{"end":112.08,"start":111.06,"word":" huge."},{"end":134.48,"start":133.88,"word":" So"},{"end":134.8,"start":134.48,"word":" now"},{"end":134.98,"start":134.8,"word":" that"},{"end":135.08,"start":134.98,"word":" we"},{"end":135.22,"start":135.08,"word":" have"},{"end":135.32,"start":135.22,"word":" an"},{"end":135.76,"start":135.32,"word":" idea"},{"end":136.18,"start":135.76,"word":" of"},{"end":136.48,"start":136.18,"word":" how"},{"end":136.78,"start":136.48,"word":" big"},{"end":137.28,"start":136.78,"word":" things"},{"end":137.78,"start":137.28,"word":" are"},{"end":138.26,"start":137.78,"word":" and"},{"end":138.44,"start":138.26,"word":" what"},{"end":138.56,"start":138.44,"word":" the"},{"end":139.02,"start":138.56,"word":" planets"},{"end":139.18,"start":139.02,"word":" look"},{"end":139.48,"start":139.18,"word":" like"},{"end":139.84,"start":139.48,"word":" compared"},{"end":140.02,"start":139.84,"word":" to"},{"end":140.22,"start":140.02,"word":" each"},{"end":140.54,"start":140.22,"word":" other,"},{"end":141.12,"start":140.9,"word":" do"},{"end":141.14,"start":141.12,"word":" you"},{"end":141.26,"start":141.14,"word":" want"},{"end":141.34,"start":141.26,"word":" to"},{"end":141.46,"start":141.34,"word":" change"},{"end":141.66,"start":141.46,"word":" your"},{"end":141.92,"start":141.66,"word":" guess?"},{"end":142.56,"start":142.38,"word":" Here's"},{"end":142.64,"start":142.56,"word":" your"},{"end":143,"start":142.64,"word":" chance."},{"end":143.64,"start":143.58,"word":" All"},{"end":143.66,"start":143.64,"word":" right."},{"end":144.16,"start":144.04,"word":" Now"},{"end":144.46,"start":144.16,"word":" let's"},{"end":144.68,"start":144.46,"word":" focus"},{"end":144.9,"start":144.68,"word":" in"},{"end":145.04,"start":144.9,"word":" on"},{"end":145.12,"start":145.04,"word":" the"},{"end":145.3,"start":145.12,"word":" two"},{"end":145.4,"start":145.3,"word":" we"},{"end":145.78,"start":145.4,"word":" really"},{"end":145.98,"start":145.78,"word":" want"},{"end":146.04,"start":145.98,"word":" to"},{"end":146.38,"start":146.04,"word":" compare."},{"end":147.06,"start":146.76,"word":" The"},{"end":147.5,"start":147.06,"word":" Sun"},{"end":148.14,"start":147.5,"word":" and"},{"end":148.38,"start":148.14,"word":" the"},{"end":148.7,"start":148.38,"word":" Earth."},{"end":149.38,"start":149.04,"word":" Now"},{"end":149.78,"start":149.38,"word":" these"},{"end":150.12,"start":149.78,"word":" are"},{"end":150.38,"start":150.12,"word":" not"},{"end":150.58,"start":150.38,"word":" to"},{"end":150.6,"start":150.58,"word":" the"},{"end":150.76,"start":150.6,"word":" right"},{"end":151.04,"start":150.76,"word":" scale."},{"end":151.38,"start":151.22,"word":" So"},{"end":151.56,"start":151.38,"word":" let's"},{"end":151.74,"start":151.56,"word":" ditch"},{"end":152.08,"start":151.74,"word":" them."},{"end":153.38,"start":153.18,"word":" And"},{"end":153.76,"start":153.38,"word":" compare"},{"end":154.18,"start":153.76,"word":" something"},{"end":154.36,"start":154.18,"word":" we"},{"end":154.72,"start":154.36,"word":" probably"},{"end":155.2,"start":154.72,"word":" all"},{"end":155.72,"start":155.2,"word":" know."},{"end":159.6,"start":159.1,"word":" Oh"},{"end":160.04,"start":159.6,"word":" basketball."},{"end":160.88,"start":160.64,"word":" If"},{"end":161.1,"start":160.88,"word":" the"},{"end":161.26,"start":161.1,"word":" Sun"},{"end":161.42,"start":161.26,"word":" were"},{"end":161.54,"start":161.42,"word":" the"},{"end":161.72,"start":161.54,"word":" size"},{"end":161.96,"start":161.72,"word":" of"},{"end":162.02,"start":161.96,"word":" a"},{"end":162.46,"start":162.02,"word":" basketball,"},{"end":163.52,"start":163.06,"word":" Earth"},{"end":164,"start":163.52,"word":" would"},{"end":164.9,"start":164,"word":" be"},{"end":165.46,"start":164.9,"word":" this"},{"end":166.1,"start":165.46,"word":" tiny"},{"end":166.44,"start":166.1,"word":" two"},{"end":166.98,"start":166.44,"word":" millimeter"},{"end":167.7,"start":166.98,"word":" bead."},{"end":168.92,"start":168.44,"word":" Right"},{"end":169.62,"start":168.92,"word":" there."},{"end":171.88,"start":171.4,"word":" All"},{"end":172.2,"start":171.88,"word":" right."},{"end":172.36,"start":172.36,"word":" Do"},{"end":172.42,"start":172.36,"word":" you"},{"end":172.68,"start":172.42,"word":" remember"},{"end":172.88,"start":172.68,"word":" your"},{"end":173.12,"start":172.88,"word":" guesses"},{"end":173.4,"start":173.12,"word":" from"},{"end":173.74,"start":173.4,"word":" earlier?"},{"end":174.02,"start":174.02,"word":" Did"},{"end":174.14,"start":174.02,"word":" you"},{"end":174.22,"start":174.14,"word":" write"},{"end":174.36,"start":174.22,"word":" them"},{"end":174.72,"start":174.36,"word":" down?"},{"end":175.06,"start":174.94,"word":" Well,"},{"end":175.36,"start":175.06,"word":" here's"},{"end":175.62,"start":175.36,"word":" one"},{"end":175.94,"start":175.62,"word":" last"},{"end":176.14,"start":175.94,"word":" time"},{"end":176.28,"start":176.14,"word":" to"},{"end":176.58,"start":176.28,"word":" update"},{"end":176.9,"start":176.58,"word":" them"},{"end":177.04,"start":176.9,"word":" if"},{"end":177.14,"start":177.04,"word":" you"},{"end":177.52,"start":177.14,"word":" want."},{"end":178.16,"start":177.9,"word":" Because"},{"end":178.28,"start":178.16,"word":" now"},{"end":178.42,"start":178.28,"word":" it's"},{"end":178.54,"start":178.42,"word":" time"},{"end":178.6,"start":178.54,"word":" to"},{"end":178.82,"start":178.6,"word":" figure"},{"end":179.24,"start":178.82,"word":" just"},{"end":179.48,"start":179.24,"word":" how"},{"end":179.72,"start":179.48,"word":" many"},{"end":180.7,"start":179.72,"word":" Earths"}],
        "context_start_s": 0,
        "context_end_s": 18,
        "context_buffer_s": 0,
        "n_search_phrases": 3,
        "n_searche_results_per_phrase": 5,
        "top_K": 5,
        "pexels_api_key": "gszGv4rBnHq1X62jwOc73uOhS0resHHzdHNPGlbU9DkjYSQtniki3bbp"
    }
}
# generate_image(job)


#%%
runpod.serverless.start({"handler": generate_image})
