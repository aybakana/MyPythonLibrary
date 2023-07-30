from diffusers import DiffusionPipeline
import torch
from PIL import Image
from IPython.display import display # Notebooks
import cv2

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


def load_refiner_model():
    pipeR = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipeR.to("cuda")
    return pipeR

def load_base_model():
    pipeB = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipeB.to("cuda")
    return pipeB

def refine_image(pipeB,pipeR,prompt,output_file):
    image = pipeB(prompt=prompt, output_type="latent").images
    image2 = pipeR(prompt=prompt, image=image).images[0]
    image2.save(output_file)

def vectorize_image(input_image_path):
    # Read the input image
    img = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a binary image
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert the contours into vector shapes (polygons)
    output_vector_image = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        output_vector_image.append(approx)
    
    return output_vector_image    
