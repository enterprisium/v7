import re
import os
import io
import json
import uuid
import torch
import pprint
import random
import requests
import numpy as np
import gradio as gr
from PIL import Image
from gtts import gTTS
from hercai import Hercai
from g4f.client import Client
import google.generativeai as genai
from TTS.api import TTS
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips, ImageClip

# Global variables for API keys
gemini_apikey = ""
elevenlabs_apikeys = ["ec71cc5fb466bbbeaa935e5a7b001d25", "675e7d7c9d7a10caf1bb77f6264cd1c9"]
segmind_apikey = ""
pexels_apikey = ""
api_key_index = 0

# Default values for the script generation
DEFAULT_TOPIC = "Success and Achievement"
DEFAULT_GOAL = "Inspire people to overcome challenges, achieve success, and celebrate their victories"
DEFAULT_LLM = "Hercai"
DEFAULT_MODEL = "V3"
DEFAULT_IMAGE_GEN = "Hercai"
DEFAULT_IMAGE_MODEL = "v3"
DEFAULT_DIMENSION = "Square"
DEFAULT_TTS = "XTTS_V2"
DEFAULT_VOICE = random.choice(["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence"])
DEFAULT_TRANSITION = "fade in"
DEFAULT_EFFECT = "zoom in"

# Voice options
ELEVENLABS_VOICES = ["Adam", "Antoni", "Arnold", "Bella", "Domi", "Elli", "Josh", "Rachel", "Sam"]
XTTS_VOICES = ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence"]

# Function to fetch image description and script
def fetch_imagedescription_and_script(prompt, llm, model=None):
    if llm == "G4F":
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()
    elif llm == "Gemini":
        if not gemini_apikey:
            raise ValueError("Gemini API key is missing! 🚨 Please add your Gemini API key to proceed.")
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
    else:
        raise ValueError("No valid JSON found in the response")

    pprint.pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]
    return {"image_prompts": image_prompts, "texts": texts, "status": "Success"}

# Function to save the API keys
def save_apikeys(gemini_key, elevenlabs_keys, segmind_key, pexels_key):
    global gemini_apikey, elevenlabs_apikeys, segmind_apikey, pexels_apikey
    gemini_apikey = gemini_key
    elevenlabs_apikeys = [key.strip() for key in elevenlabs_keys.split(',')]
    segmind_apikey = segmind_key
    pexels_apikey = pexels_key

    # Check remaining characters for ElevenLabs API keys
    total_remaining_characters = 0
    for key in elevenlabs_apikeys:
        url = "https://api.elevenlabs.io/v1/user"
        headers = {"xi-api-key": key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            remaining_characters = data.get("subscription", {}).get("character_limit", 0) - data.get("subscription", {}).get("character_count", 0)
            total_remaining_characters += remaining_characters
        else:
            return f"Error checking API key {key}: {response.text}"

    return f"API keys saved successfully! Total remaining characters for ElevenLabs: {total_remaining_characters}"

# Function to generate images
def generate_images(prompts, active_folder, image_gen, image_model, dimension):
    if not os.path.exists(active_folder):
        os.makedirs(active_folder)

    image_logs = []
    image_files = []

    # Determine the dimensions based on the selected option
    if dimension == "Landscape":
        img_width, img_height = 1920, 1080
    elif dimension == "Portrait":
        img_width, img_height = 1080, 1920
    elif dimension == "Square":
        img_width, img_height = 1080, 1080
    else:
        img_width, img_height = 1024, 1024  # Default dimensions

    if image_gen == "Hercai":
        herc = Hercai("")
        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), {img_width}x{img_height}, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
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
        if not segmind_apikey:
            raise ValueError("Segmind API key is missing! 🚨 Please add your Segmind API key to proceed.")
        url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
        headers = {'x-api-key': segmind_apikey}

        num_images = len(prompts)
        currentseed = random.randint(1, 1000000)
        print("seed ", currentseed)

        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), {img_width}x{img_height}, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
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
                "img_width": img_width,
                "img_height": img_height,
                "refiner": "yes",
                "base64": False
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
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
    elif image_gen == "Pexels":
        if not pexels_apikey:
            raise ValueError("Pexels API key is missing! 🚨 Please add your Pexels API key to proceed.")
        url = "https://api.pexels.com/v1/search"
        headers = {'Authorization': pexels_apikey}

        for i, prompt in enumerate(prompts):
            params = {
                "query": prompt,
                "per_page": 1
            }
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                if data["photos"]:
                    image_url = data["photos"][0]["src"]["original"]
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image_data = image_response.content
                        image = Image.open(io.BytesIO(image_data))
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
                else:
                    log_message = f"No images found for prompt {i + 1}"
                    print(log_message)
                    image_logs.append(log_message)
            else:
                log_message = f"Error: Failed to retrieve image {i + 1}"
                print(log_message)
                image_logs.append(log_message)

    return {"logs": image_logs, "folder": active_folder, "files": image_files}

# Function to generate speech
def generate_speech(texts, tts, voice, speech_volume, bgm_volume, foldername="output"):
    if tts == "Elevenlabs":
        if not elevenlabs_apikeys:
            raise ValueError("ElevenLabs API keys are missing! 🚨 Please add your ElevenLabs API keys to proceed.")
        voice_id = "pNInz6obpgDQGcFmaJgB"  # Example voice ID for Elevenlabs

        def generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
            global api_key_index  # Access the global index variable

            # Cycle through API keys
            api_key = elevenlabs_apikeys[api_key_index]
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }

            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 429:  # Handle quota exceeded error
                print("Quota exceeded for current API key. Switching to the next key.")
                api_key_index = (api_key_index + 1) % len(elevenlabs_apikeys)
                if api_key_index == 0:
                    print("All ElevenLabs API keys exhausted. Falling back to gTTS.")
                    generate_speech_with_gtts(text, foldername, filename)
                else:
                    generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id, stability, similarity_boost)  # Retry with the new key
            elif response.status_code != 200:
                print(response.text)
            else:
                file_path = f"{foldername}/{filename}.mp3"
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Text: {text} -> Converted to: {file_path}")

        def generate_speech_with_gtts(text, foldername, filename):
            tts = gTTS(text=text, lang='en')
            file_path = f"{foldername}/{filename}.mp3"
            tts.save(file_path)
            print(f"Text: {text} -> Converted to: {file_path} using gTTS")

        for i, text in enumerate(texts):
            output_filename = str(i + 1)
            generate_speech_with_elevenlabs(text, foldername, output_filename, voice_id)

    elif tts == "XTTS_V2":
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)

        for i, text in enumerate(texts):
            path = f"{foldername}/{i + 1}.mp3"
            tts.tts_to_file(text=text, file_path=path, language="en", speaker=voice)
            print(f"Text: {text} -> Converted to: {path}")

    else:
        print("Invalid TTS provider choice. Please choose either 'Elevenlabs' or 'XTTS_V2'.")

# Function to generate the script, images, and speech
def generate_script_images_and_speech(topic, goal, llm, model, image_gen, image_model, dimension, tts, voice, speech_volume, bgm_volume):
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

    try:
        result = fetch_imagedescription_and_script(prompt, llm, model)
        if result["status"] == "Success":
            image_prompts = result["image_prompts"]
            texts = result["texts"]
            active_folder = str(uuid.uuid4())
            image_result = generate_images(image_prompts, active_folder, image_gen, image_model, dimension)
            generate_speech(texts, tts, voice, speech_volume, bgm_volume, foldername=active_folder)
            return {"script": result, "image_result": image_result, "status": "Success"}, image_result["files"]
        else:
            return {"error": result["error"], "status": "Failed"}, []
    except Exception as e:
        return {"error": str(e), "status": "Failed"}, []

# Gradio interface with base theme
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🚀 SocialGPT: AI-Powered Short Video Generator")

    with gr.Accordion("🔑 API Key Settings", open=False):
        with gr.Row():
            gemini_apikey_input = gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here", type="password")
            elevenlabs_apikey_input = gr.Textbox(label="ElevenLabs API Keys (comma-separat)", placeholder="Enter your ElevenLabs API keys here", type="password")
            segmind_apikey_input = gr.Textbox(label="Segmind API Key", placeholder="Enter your Segmind API key here", type="password")
            pexels_apikey_input = gr.Textbox(label="Pexels API Key", placeholder="Enter your Pexels API key here", type="password")
        save_apikey_btn = gr.Button("💾 Save API Keys", variant="primary")
        save_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        topic = gr.Textbox(label="Topic", placeholder="Enter the topic for the video", value=DEFAULT_TOPIC)
        goal = gr.Textbox(label="Goal", placeholder="Enter the goal for the video", value=DEFAULT_GOAL)

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            LLM = gr.Radio(["Gemini", "Hercai", "G4F"], label="Select Your LLM", value=DEFAULT_LLM)
            model = gr.Dropdown(label="Select Model (Hercai only)", choices=["V3", "V3-32k", "Turbo", "Turbo-16k", "Gemini", "LLama3-70b", "LLama3-8b", "Mixtral-8x7b", "Gemma-7b", "Gemma2-9b"], value=DEFAULT_MODEL, visible=True)

        with gr.Column(scale=1, min_width=200):
            image_gen = gr.Radio(label="Video Source", choices=["Segmind", "Hercai", "Pexels"], value=DEFAULT_IMAGE_GEN)
            image_model = gr.Dropdown(label="Image Model (Hercai only)", choices=["v1", "v2", "v2-beta", "v3", "lexica", "prodia", "simurg", "animefy", "raava", "shonin"], value=DEFAULT_IMAGE_MODEL, visible=True)

        with gr.Column(scale=1, min_width=200):
            tts = gr.Radio(label="Audio Source", choices=["Elevenlabs", "XTTS_V2", "GTTS"], value=DEFAULT_TTS)
            voice = gr.Dropdown(label="Voice", choices=XTTS_VOICES, value=DEFAULT_VOICE)

    with gr.Accordion("Others", open=False):
        with gr.Row():
            speech_volume = gr.Slider(label="Speech Volume", minimum=0, maximum=100, value=50)
            bgm_volume = gr.Slider(label="Background Music Volume", minimum=0, maximum=100, value=50)
            dimension = gr.Dropdown(label="Dimension", choices=["Landscape", "Portrait", "Square"], value=DEFAULT_DIMENSION)
            transition = gr.Dropdown(label="Transition", choices=["fade in", "fade out"], value=DEFAULT_TRANSITION)
            effect = gr.Dropdown(label="Effect", choices=["zoom in", "zoom out"], value=DEFAULT_EFFECT)

        with gr.Row():
            watermark_checkbox = gr.Checkbox(label="Watermark", value=False)
            watermark_text = gr.Textbox(label="Watermark Text", placeholder="Enter watermark text", visible=False)
            subtitle_font = gr.Dropdown(label="Font", choices=[f for f in os.listdir("Fonts") if f.endswith(".ttf")], value="Arial")
            subtitle_color = gr.ColorPicker(label="Subtitle Color", value="#FFFFFF")
            active_word_color = gr.ColorPicker(label="Active Word Color", value="#FFFF00")

        with gr.Row():
            outline_color = gr.ColorPicker(label="Outline Color", value="#000000")
            outline_width = gr.Slider(label="Outline Width", minimum=0, maximum=10, value=2)
            stroke = gr.Slider(label="Stroke", minimum=0, maximum=10, value=1)
            subtitle_position = gr.Dropdown(label="Subtitle Position", choices=["top", "center", "bottom"], value="bottom")

    with gr.Row():
        cancel_btn = gr.Button("❌ Cancel", variant="secondary")
        generate_btn = gr.Button("🎬 Generate", variant="primary")
        reset_btn = gr.Button("🔄 Reset", variant="secondary")

    output_video = gr.Video(label="Generated Video")

    def update_model_dropdown(llm):
        return gr.update(visible=llm == "Hercai")

    def update_image_model_dropdown(image_gen):
        return gr.update(visible=image_gen == "Hercai")

    def update_voice_dropdown(tts):
        if tts == "Elevenlabs":
            return gr.update(choices=ELEVENLABS_VOICES, value=random.choice(ELEVENLABS_VOICES))
        elif tts == "XTTS_V2":
            return gr.update(choices=XTTS_VOICES, value=random.choice(XTTS_VOICES))
        else:
            return gr.update(choices=[], value="")

    def update_watermark_text_visibility(watermark):
        return gr.update(visible=watermark)

    def handle_generate_script_images_and_speech(topic, goal, llm, model, image_gen, image_model, dimension, tts, voice, speech_volume, bgm_volume, transition, effect, watermark, watermark_text, subtitle_font, subtitle_color, active_word_color, outline_color, outline_width, stroke, subtitle_position):
        try:
            if not topic:
                raise ValueError("Please provide the Topic")
            if not goal:
                raise ValueError("Please provide the Goal")

            result, files = generate_script_images_and_speech(topic, goal, llm, model, image_gen, image_model, dimension, tts, voice, speech_volume, bgm_volume)
            if result["status"] == "Success":
                subtitle_settings = {
                    "font": subtitle_font,
                    "color": subtitle_color,
                    "active_word_color": active_word_color,
                    "outline_color": outline_color,
                    "outline_width": outline_width,
                    "stroke": stroke,
                    "position": subtitle_position
                }
                video_path = create_final_video(result["image_result"]["folder"], transition, effect, watermark_text if watermark else None, subtitle_settings)
                return video_path, gr.update(visible=False)
            else:
                raise ValueError(result["error"])
        except Exception as e:
            error_message = f"<div style='color: red; font-size: 20px;'>🚨 Oops! Something went wrong: {str(e)} 😱</div>"
            return gr.update(visible=False), gr.update(value=error_message, visible=True)

    def handle_reset():
        return (DEFAULT_TOPIC, DEFAULT_GOAL, DEFAULT_LLM, DEFAULT_MODEL, DEFAULT_IMAGE_GEN, DEFAULT_IMAGE_MODEL, DEFAULT_DIMENSION, DEFAULT_TTS, DEFAULT_VOICE, 50, 50, DEFAULT_TRANSITION, DEFAULT_EFFECT, False, "", "Arial", "#FFFFFF", "#FFFF00", "#000000", 2, 1, "bottom")

    LLM.change(update_model_dropdown, inputs=LLM, outputs=model)
    image_gen.change(update_image_model_dropdown, inputs=image_gen, outputs=image_model)
    tts.change(update_voice_dropdown, inputs=tts, outputs=voice)
    watermark_checkbox.change(update_watermark_text_visibility, inputs=watermark_checkbox, outputs=watermark_text)
    generate_btn.click(handle_generate_script_images_and_speech,
                       inputs=[topic, goal, LLM, model, image_gen, image_model, dimension, tts, voice, speech_volume, bgm_volume, transition, effect, watermark_checkbox, watermark_text, subtitle_font, subtitle_color, active_word_color, outline_color, outline_width, stroke, subtitle_position],
                       outputs=[output_video, gr.HTML()])
    reset_btn.click(handle_reset, outputs=[topic, goal, LLM, model, image_gen, image_model, dimension, tts, voice, speech_volume, bgm_volume, transition, effect, watermark_checkbox, watermark_text, subtitle_font, subtitle_color, active_word_color, outline_color, outline_width, stroke, subtitle_position])
    save_apikey_btn.click(save_apikeys, inputs=[gemini_apikey_input, elevenlabs_apikey_input, segmind_apikey_input, pexels_apikey_input], outputs=save_status)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
