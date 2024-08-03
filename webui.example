#THIS IS JUST GRADIO EXAMPLE 
#THIS IS JUST WEBUI EXAMPLE
#THIS HAS 3 CLOUMS


import gradio as gr

def dummy_function(*args):
    return "Operation completed"

with gr.Blocks(theme=gr.themes.Base(
    primary_hue="blue",
    secondary_hue="orange",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)) as demo:
    gr.Markdown("# Advanced RVC Inference")
    
    with gr.Row():
        with gr.Column(scale=3):
            weight = gr.Dropdown(label="Weight", choices=[])
        with gr.Column(scale=3):
            index_file = gr.Dropdown(label="List of index file", choices=[])
        with gr.Column(scale=2):
            refresh_btn = gr.Button("Refresh model list")
        with gr.Column(scale=2):
            clear_btn = gr.Button("Clear Model from memory")
    
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.Markdown("### No model selected")
            
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    input_voice = gr.Audio(label="Input voice", type="filepath")
                    gr.Markdown("Drop Audio Here\n- or -\nClick to Upload")
                    use_mic = gr.Checkbox(label="Use Microphone")
                    upload_file = gr.File(label="Upload audio file")
                    splitter_model = gr.Dropdown(label="Splitter Model", choices=["htdemucs"], value="htdemucs")
                    output_info_1 = gr.Textbox(label="Output Information")
                    split_btn = gr.Button("Split Audio", variant="primary")
                    vocal_preview = gr.Audio(label="Vocal Preview")
                
                with gr.Column(scale=1, min_width=200):
                    transpose = gr.Number(label="Transpose", value=0)
                    pitch_algorithm = gr.Radio(["pm", "harvest", "rmvpe", "crepe"], label="Pitch extraction algorithm", value="pm")
                    feature_ratio = gr.Slider(0, 1, value=0.7, label="Retrieval feature ratio")
                    median_filtering = gr.Slider(0, 10, value=3, step=1, label="Apply Median Filtering")
                    resample = gr.Slider(0, 48000, value=0, step=1, label="Resample the output audio")
                    volume_envelope = gr.Slider(0, 1, value=1, label="Volume Envelope")
                    voice_protection = gr.Slider(0, 1, value=0.5, label="Voice Protection")
                
                with gr.Column(scale=1, min_width=200):
                    output_info_2 = gr.Textbox(label="Output Information")
                    output_audio = gr.Audio(label="Output Audio")
                    convert_btn = gr.Button("Convert", variant="primary")
                    vocal_volume = gr.Slider(0, 2, value=1, label="Vocal volume")
                    instrument_volume = gr.Slider(0, 2, value=1, label="Instrument volume")
                    combined_audio = gr.Audio(label="Output Combined Audio")
                    combine_btn = gr.Button("Combine", variant="primary")

        gr.TabItem("Batch Inference")
        gr.TabItem("Model Downloader")
        gr.TabItem("Settings")

    split_btn.click(dummy_function, inputs=[input_voice], outputs=[output_info_1, vocal_preview])
    convert_btn.click(dummy_function, inputs=[input_voice, transpose, pitch_algorithm, feature_ratio, median_filtering, resample, volume_envelope, voice_protection], outputs=[output_info_2, output_audio])
    combine_btn.click(dummy_function, inputs=[vocal_volume, instrument_volume], outputs=combined_audio)

demo.launch(share=True)
