#!pip install transformers==4.25.1 diffusers==0.12.1 invisible-watermark accelerate spacy ftfy safetensors scipy
# model_id = "runwayml/stable-diffusion-v1-5"
from diffusers import StableDiffusionPipeline
import torch

def load_model(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def load_stable_diffusion_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe    

def load_dreamlike_model():
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    pipeD = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeD = pipeD.to("cuda")
    return pipeD

# when pipe, prompt and output file path are given, image is generated and saved at the output file path
def generate_image(pipe, prompt,output_file):
    image = pipe(prompt).images[0]
    image.save(output_file)

import random

# when a keyword list, min and max number of keywords are given, a random prompt is generated
def generate_random_prompts(keyword_list, min_keywords=10, max_keywords=20):
    num_keywords = random.randint(min_keywords, max_keywords)
    selected_keywords = random.sample(keyword_list, num_keywords)
    prompt = ", ".join(selected_keywords)
    return prompt

def save_log_drive(prompt, output_file, seed,log_file):
    text = f"{prompt,output_file,seed}"

def set_deterministic():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
