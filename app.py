!apt install -q imagemagick
!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml

!pip install --quiet hercai
!pip install --quiet elevenlabs
!pip install --quiet g4f[all] --upgrade
!pip install --quiet gradio==3.41.2
!pip install --quiet faster-whisper==0.7.0
!pip install --quiet imageio==2.25.1
!pip install --quiet ffmpeg-python==0.2.0
!pip install --quiet git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
!pip install --quiet TTS



gemini_apikey = "AIzaSyC6N1MVe9WmAFjWMNuXjlaLnYa8e3tY"

import re
import os
import io
import json
import time
import uuid
import random
import pprint
import requests
import ffmpeg
import numpy as np
import gradio as gr
from PIL import Image
from hercai import Hercai
from elevenlabs import play
from g4f.client import Client
from typing import Tuple, List
import google.generativeai as genai
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips, ImageClip


LLM = "Gemini"  # @param ["Gemini", "G4F", "Hercai"] {allow-input: true}
MODEL = "turbo"  # @param ["v3", "v3-32k", "turbo", "turbo-16k", "gemini", "llama3-70b", "llama3-8b", "mixtral-8x7b", "gemma-7b", "gemma2-9b"]

def fetch_imagedescription_and_script(prompt, llm, model=None):
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


    # Extract JSON from the response
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        json_str = json_match.group(0)
        # Ensure the JSON array is properly closed
        if not json_str.endswith(']'):
            json_str += ']'
        output = json.loads(json_str)
    else:
        raise ValueError("No valid JSON found in the response")

    pprint.pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]
    return image_prompts, texts

topic = "Success and Achievement"
goal = "Inspire people to overcome challenges, achieve success, and celebrate their victories"

prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic, goal)

sample_output = """
   [
       { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
       { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
       ...
   ]"""

prompt_postinstruction = """By following these instructions, you will create an impactful {} short-form video.
Output:""".format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

image_prompts, texts = fetch_imagedescription_and_script(prompt, LLM, MODEL)
print("image_prompts: ", image_prompts)
print("texts: ", texts)
print(len(texts))


import uuid

current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print (active_folder)

IMAGE_GEN = "Hercai" # @param ["Segmind", "Hercai"] {allow-input: true}

# image model only works if IMAGE_GEN is Hercai
IMAGE_MODEL = "simurg" # @param ["v1", "v2", "v2-beta", "v3", "lexica", "prodia", "simurg", "animefy", "raava", "shonin"] {allow-input: true}


def generate_images(prompts, active_folder, image_gen, image_model=IMAGE_MODEL):
    if not os.path.exists(active_folder):
        os.makedirs(active_folder)

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
                    image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                    image.save(image_filename)
                    print(f"Image {i + 1}/{len(prompts)} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to download image {i + 1}")
            except Exception as e:
                print(f"Error generating image {i + 1}: {str(e)}")
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
                image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                image.save(image_filename)
                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
            else:
                print(response.text)
                print(f"Error: Failed to retrieve or save image {i + 1}")

generate_images(image_prompts, active_folder, IMAGE_GEN)


TTS = "XTTS_V2" # @param ["Elevenlabs", "XTTS_V2"] {allow-input: true}
LANGUAGE = "en" # @param ["en", "English", "Chinese", "Spanish", "Hindi", "Portuguese", "French", "German", "Japanese", "Arabic", "Korean", "Italian", "Dutch", "Turkish", "Polish", "Russian", "Czech"]
VOICE = "Baldur Sanjin" # @param ["Sarah", "Laura", "Charlie", "George", "Callum", "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel", "Lily", "Bill", "Claribel Dervla", "Daisy Studious", "Tammie Ema"]

api_keys = os.environ.get('ELEVENLABS_API_KEYS', '675e7d7c9d7a10caf1bb77f6264cd1c9,sk_7659d722c316eb37935ec20cedf13cea5e80dd5ac899b95f').split(',')
api_key_index = 0

import os
import requests
import torch
from TTS.api import TTS as TTS_API  # Rename imported TTS to avoid conflict

api_keys = os.environ.get('ELEVENLABS_API_KEYS', '675e7d7c9d7a10caf1bb77f6264cd1c9,sk_7659d722c316eb37935ec20cedf13cea5e80dd5ac899b95f').split(',')
api_key_index = 0

def generate_speech(texts, TTS="XTTS_V2", language="en", voice="Badr Odhiambo", foldername="output"):
    if TTS == "Elevenlabs":
        voice_id = "pNInz6obpgDQGcFmaJgB"

        def generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
            global api_key_index  # Access the global index variable

            # Cycle through API keys
            api_key = api_keys[api_key_index]
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
                api_key_index = (api_key_index + 1) % len(api_keys)
                generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id, stability, similarity_boost)  # Retry with the new key
            elif response.status_code != 200:
                print(response.text)
            else:
                file_path = f"{foldername}/{filename}.mp3"
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Text: {text} -> Converted to: {file_path}")

        for i, text in enumerate(texts):
            output_filename = str(i + 1)
            generate_speech_with_elevenlabs(text, foldername, output_filename, voice_id)

    elif TTS == "XTTS_V2":
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS_API("tts_models/multilingual/multi-dataset/xtts_v2")  # Use TTS_API to initialize
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)

        for i, text in enumerate(texts):
            path = f"{active_folder}/{i + 1}.mp3"  # Assuming 'active_folder' is defined elsewhere
            tts.tts_to_file(text=text, file_path=path, language=language, speaker=voice)
            print(f"Text: {text} -> Converted to: {path}")

    else:
        print("Invalid TTS provider choice. Please choose either 'Elevenlabs' or 'XTTS_V2'.")
# Generate speech with the chosen provider, language, and voice
generate_speech(texts, TTS=TTS, language=LANGUAGE, voice=VOICE)

"""## 4. Combine All Previously Generated Images and Speech into Video"""

from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
import os
import cv2
import numpy as np

def create_combined_video_audio(mp3_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    mp3_files = sorted([file for file in os.listdir(mp3_folder) if file.endswith(".mp3")])
    mp3_files = sorted(mp3_files, key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(os.path.join(mp3_folder, mp3_file))
        audio_clips.append(audio_clip)

        # Load the corresponding image for each mp3 and set its duration to match the mp3's duration
        img_path = os.path.join(mp3_folder, f"{mp3_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

        # Resize the original image to 1080x1080
        image_resized = cv2.resize(image, (1080, 1080))

        # Blur the image
        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - 1080) // 2
        blurred_img[y_offset:y_offset+1080, :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(final_audio)
    finalpath = mp3_folder+"/"+output_filename

    final_video.write_videofile(finalpath, fps=fps, codec='libx264',audio_codec="aac")

output_filename = "combined_video.mp4"
create_combined_video_audio(active_folder, output_filename)

def extract_audio_from_video(outvideo):
    """
    Extract audio from a video file and save it as an MP3 file.

    :param output_video_file: Path to the video file.
    :return: Path to the generated audio file.
    """

    audiofilename = outvideo.replace(".mp4",'.mp3')

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(outvideo)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename



audiofilename = extract_audio_from_video(output_video_file)
print(audiofilename)

from IPython.display import Audio

Audio(audiofilename)

from faster_whisper import WhisperModel

model_size = "base"
model = WhisperModel(model_size)

segments, info = model.transcribe(audiofilename, word_timestamps=True)
segments = list(segments)  # The transcription will actually run here.
for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

wordlevel_info = []

for segment in segments:
    for word in segment.words:
      wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})

wordlevel_info

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load the video file
video = VideoFileClip(output_video_file)

# Function to generate text clips
def generate_text_clip(word, start, end, video):
    txt_clip = (TextClip(word,font_size=80,color='white',font = "Nimbus-Sans-Bold",stroke_width=3, stroke_color='black').with_position('center')
               .with_duration(end - start))

    return txt_clip.with_start(start)

# Generate a list of text clips based on timestamps
clips = [generate_text_clip(item['word'], item['start'], item['end'], video) for item in wordlevel_info]

# Overlay the text clips on the video
final_video = CompositeVideoClip([video] + clips)

finalvideoname = active_folder+"/"+"final.mp4"
# Write the result to a file
final_video.write_videofile(finalvideoname, codec="libx264",audio_codec="aac")
