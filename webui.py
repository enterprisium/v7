import gradio as gr
import json
import re
import pprint
from hercai import Hercai
from g4f.client import Client
import google.generativeai as genai
import os
import io
import requests
from PIL import Image
import uuid
import cv2
import numpy as np
import random

# Global variables for API keys
gemini_apikey = ""
elevenlabs_apikey = ""
segmind_apikey = ""

# Default values for the script generation
DEFAULT_TOPIC = "Success and Achievement"
DEFAULT_GOAL = "Inspire people to overcome challenges, achieve success, and celebrate their victories"
DEFAULT_LLM = "G4F"
DEFAULT_IMAGE_GEN = "Hercai"
DEFAULT_IMAGE_MODEL = "v2"
DEFAULT_DIMENSION = "Landscape"

# Function to fetch image description and script
def fetch_imagedescription_and_script(prompt, llm, model=None):
    while True:
        try:
            if llm == "G4F":
                client = Client()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content.strip()
            elif llm == "Gemini":
                genai.configure(api_key=gemini_apikey)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                response_text = response.text
            elif llm == "Hercai":
                herc = Hercai("")
                response = herc.question(model=model, content=prompt)
                response_text = response["reply"]
            else:
                raise ValueError("Invalid LLM selected")

            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group(0)
                if not json_str.endswith(']'):
                    json_str += ']'
                output = json.loads(json_str)
                break
            else:
                raise ValueError("No valid JSON found in the response")
        except Exception as e:
            print(f"Error: {str(e)}. Retrying...")

    pprint.pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]
    return {"image_prompts": image_prompts, "texts": texts, "status": "Success"}

# Function to save the API keys
def save_apikeys(gemini_key, elevenlabs_key, segmind_key):
    global gemini_apikey, elevenlabs_apikey, segmind_apikey
    gemini_apikey = gemini_key
    elevenlabs_apikey = elevenlabs_key
    segmind_apikey = segmind_key
    return "API keys saved successfully!"

# Function to generate images
def generate_images(prompts, active_folder, image_gen, image_model, dimension):
    if not os.path.exists(active_folder):
        os.makedirs(active_folder)

    image_logs = []
    image_files = []

    if image_gen == "Hercai":
        herc = Hercai("")
        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
            try:
                image_result = herc.draw_image(
                    model=image_model,
                    prompt=final_prompt,
                    negative_prompt="((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs"
                )
                image_url = image_result["url"]
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image = process_image(image, dimension)
                    image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                    image.save(image_filename)
                    log_message = f"Image {i + 1}/{len(prompts)} saved as '{image_filename}'"
                    print(log_message)
                    image_logs.append(log_message)
                    image_files.append(image_filename)
                else:
                    log_message = f"Error: Failed to download image {i + 1}"
                    print(log_message)
                    image_logs.append(log_message)
            except Exception as e:
                log_message = f"Error generating image {i + 1}: {str(e)}"
                print(log_message)
                image_logs.append(log_message)
    elif image_gen == "Segmind":
        url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
        headers = {'x-api-key': segmind_apikey}

        num_images = len(prompts)
        currentseed = random.randint(1, 1000000)
        print("seed ", currentseed)

        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
            data = {
                "prompt": final_prompt,
                "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
                "style": "hdr",
                "samples": 1,
                "scheduler": "UniPC",
                "num_inference_steps": 30,
                "guidance_scale": 8,
                "strength": 1,
                "seed": currentseed,
                "img_width": 1024,
                "img_height": 1024,
                "refiner": "yes",
                "base64": False
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
                image = process_image(image, dimension)
                image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                image.save(image_filename)
                log_message = f"Image {i + 1}/{num_images} saved as '{image_filename}'"
                print(log_message)
                image_logs.append(log_message)
                image_files.append(image_filename)
            else:
                print(response.text)
                log_message = f"Error: Failed to retrieve or save image {i + 1}"
                print(log_message)
                image_logs.append(log_message)

    return {"logs": image_logs, "folder": active_folder, "files": image_files}

# Function to process image based on dimension
def process_image(image, dimension):
    if dimension == "Landscape":
        return image.resize((1920, 1080))
    elif dimension == "Portrait":
        return image.resize((1080, 1920))
    elif dimension == "Square":
        return image.resize((1080, 1080))
    return image

# Function to generate the script and images
def generate_script_and_images(topic, goal, llm, model, image_gen, image_model, dimension):
    prompt_prefix = f"""You are tasked with creating a script for a {topic} video that is about 30 seconds.
    Your goal is to {goal}.
    Please follow these instructions to create an engaging and impactful video:
    1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
    2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
    3. For each scene cut, provide a detailed description of the stock image being shown.
    4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
    5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
    6. Strictly output your response in a JSON list format, adhering to the following sample structure:"""

    sample_output = """
       [
           { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
           { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
           ...
       ]"""

    prompt_postinstruction = f"""By following these instructions, you will create an impactful {topic} short-form video.
    Output:"""

    prompt = prompt_prefix + sample_output + prompt_postinstruction

    result = fetch_imagedescription_and_script(prompt, llm, model)
    if result["status"] == "Success":
        image_prompts = result["image_prompts"]
        texts = result["texts"]
        active_folder = str(uuid.uuid4())
        image_result = generate_images(image_prompts, active_folder, image_gen, image_model, dimension)
        return {"script": result, "image_result": image_result, "status": "Success"}
    else:
        return {"error": result["error"], "status": "Failed"}

# Custom theme
custom_theme = gr.themes.Soft().set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    button_primary_background_fill="*secondary_600",
    button_primary_background_fill_dark="*secondary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
)

# Gradio interface
with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# 🚀 SocialGPT: AI-Powered Short Video Generator")
    
    with gr.Accordion("🔑 API Key Settings", open=False):
        with gr.Row():
            gemini_apikey_input = gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here", type="password")
            elevenlabs_apikey_input = gr.Textbox(label="ElevenLabs API Key", placeholder="Enter your ElevenLabs API key here", type="password")
            segmind_apikey_input = gr.Textbox(label="Segmind API Key", placeholder="Enter your Segmind API key here", type="password")
        
        with gr.Row():
            save_apikey_btn = gr.Button("💾 Save API Keys", variant="primary")
            save_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Textbox(label="Topic", value=DEFAULT_TOPIC, placeholder="Enter the topic here")
            goal = gr.Textbox(label="Goal", value=DEFAULT_GOAL, placeholder="Enter the goal here")
            llm = gr.Dropdown(label="Select LLM", choices=["G4F", "Gemini", "Hercai"], value=DEFAULT_LLM)
            model = gr.Dropdown(label="Select Model (Hercai only)", choices=["v3", "v3-32k", "turbo", "turbo-16k", "gemini", "llama3-70b", "llama3-8b", "mixtral-8x7b", "gemma-7b", "gemma2-9b"], visible=True if DEFAULT_LLM == "Hercai" else False)
        with gr.Column(scale=1):
            image_gen = gr.Dropdown(label="Video Source", choices=["Segmind", "Hercai"], value=DEFAULT_IMAGE_GEN)
            image_model = gr.Dropdown(label="Image Model (Hercai only)", choices=["v1", "v2", "v2-beta", "v3", "lexica", "prodia", "simurg", "animefy", "raava", "shonin"], value=DEFAULT_IMAGE_MODEL, visible=True if DEFAULT_IMAGE_GEN == "Hercai" else False)
            dimension = gr.Dropdown(label="Dimension", choices=["Landscape", "Portrait", "Square"], value=DEFAULT_DIMENSION)
    
    generate_btn = gr.Button("🎬 Generate Video", variant="primary")
    output = gr.JSON(label="Generated Result")
    image_gallery = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto")

    def update_model_dropdown(llm):
        return gr.update(visible=llm == "Hercai")

    def update_image_model_dropdown(image_gen):
        return gr.update(visible=image_gen == "Hercai")

    def display_images(result):
        if result["status"] == "Success":
            return result["image_result"]["files"]
        return []

    llm.change(update_model_dropdown, inputs=llm, outputs=model)
    image_gen.change(update_image_model_dropdown, inputs=image_gen, outputs=image_model)
    generate_btn.click(generate_script_and_images, 
                       inputs=[topic, goal, llm, model, image_gen, image_model, dimension], 
                       outputs=[output, image_gallery])
    save_apikey_btn.click(save_apikeys, inputs=[gemini_apikey_input, elevenlabs_apikey_input, segmind_apikey_input], outputs=save_status)

    gr.Markdown("""
    ## 🌟 How to Use SocialGPT
    1. Set up your API keys in the 'API Key Settings' section.
    2. Choose a topic and goal for your video.
    3. Select your preferred Language Model (LLM) and Image Generation source.
    4. Choose the video dimension (Landscape, Portrait, or Square).
    5. Click 'Generate Video' to create your script and images.
    6. The generated result will show the script and image generation status.
    7. Generated images will be displayed in the gallery below.

    Work in Progress!
    """)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
