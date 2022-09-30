import io
import base64
import os

import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from PIL import ImageOps
import gradio as gr
import base64
import skimage
from utils import *


def load_html():
    body, canvaspy = "", ""
    with open("index.html", "r") as f:
        body = f.read()
    with open("canvas.py", "r") as f:
        canvaspy = f.read()
    body = body.replace("- paths:\n", "")
    body = body.replace("  - ./canvas.py\n", "")
    body = body.replace("from canvas import InfCanvas", canvaspy)
    return body


def test(x):
    x = load_html()
    return f"""<iframe id="sdinfframe" style="width: 100%; height: 700px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


DEBUG_MODE = False

try:
    SAMPLING_MODE = Image.Resampling.LANCZOS
except Exception as e:
    SAMPLING_MODE = Image.LANCZOS

try:
    contain_func = ImageOps.contain
except Exception as e:

    def contain_func(image, size, method=SAMPLING_MODE):
        # from PIL: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.contain
        im_ratio = image.width / image.height
        dest_ratio = size[0] / size[1]
        if im_ratio != dest_ratio:
            if im_ratio > dest_ratio:
                new_height = int(image.height / image.width * size[0])
                if new_height != size[1]:
                    size = (size[0], new_height)
            else:
                new_width = int(image.width / image.height * size[1])
                if new_width != size[0]:
                    size = (new_width, size[1])
        return image.resize(size, resample=method)


PAINT_SELECTION = "✥"
IMAGE_SELECTION = "🖼️"
BRUSH_SELECTION = "🖌️"
blocks = gr.Blocks()
model = {}


def get_token():
    token = ""
    if os.path.exists(".token"):
        with open(".token", "r") as f:
            token = f.read()
    token = os.environ.get("hftoken", token)
    return token


def save_token(token):
    with open(".token", "w") as f:
        f.write(token)


def get_model(token=""):
    if "text2img" not in model:
        text2img = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=token,
        ).to("cuda")
        model["safety_checker"] = text2img.safety_checker
        inpaint = StableDiffusionInpaintPipeline(
            vae=text2img.vae,
            text_encoder=text2img.text_encoder,
            tokenizer=text2img.tokenizer,
            unet=text2img.unet,
            scheduler=text2img.scheduler,
            safety_checker=text2img.safety_checker,
            feature_extractor=text2img.feature_extractor,
        ).to("cuda")
        save_token(token)
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory // (
                1024 ** 3
            )
            if total_memory <= 5:
                inpaint.enable_attention_slicing()
        except:
            pass
        model["text2img"] = text2img
        model["inpaint"] = inpaint
    return model["text2img"], model["inpaint"]


def run_outpaint(
    sel_buffer_str,
    prompt_text,
    strength,
    guidance,
    step,
    resize_check,
    fill_mode,
    enable_safety,
    state,
):
    base64_str = "base64"
    if True:
        text2img, inpaint = get_model()
        if enable_safety:
            text2img.safety_checker = model["safety_checker"]
            inpaint.safety_checker = model["safety_checker"]
        else:
            text2img.safety_checker = lambda images, **kwargs: (images, False)
            inpaint.safety_checker = lambda images, **kwargs: (images, False)
        data = base64.b64decode(str(sel_buffer_str))
        pil = Image.open(io.BytesIO(data))
        # base.output.clear_output()
        # base.read_selection_from_buffer()
        sel_buffer = np.array(pil)
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        process_size = 512 if resize_check else model["sel_size"]
        if mask.sum() > 0:
            img, mask = functbl[fill_mode](img, mask)
            init_image = Image.fromarray(img)
            mask = 255 - mask
            mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
            mask = mask.repeat(8, axis=0).repeat(8, axis=1)
            mask_image = Image.fromarray(mask)
            # mask_image=mask_image.filter(ImageFilter.GaussianBlur(radius = 8))
            with autocast("cuda"):
                images = inpaint(
                    prompt=prompt_text,
                    init_image=init_image.resize(
                        (process_size, process_size), resample=SAMPLING_MODE
                    ),
                    mask_image=mask_image.resize((process_size, process_size)),
                    strength=strength,
                    num_inference_steps=step,
                    guidance_scale=guidance,
                )["sample"]
        else:
            with autocast("cuda"):
                images = text2img(
                    prompt=prompt_text, height=process_size, width=process_size,
                )["sample"]
        out = sel_buffer.copy()
        out[:, :, 0:3] = np.array(
            images[0].resize(
                (model["sel_size"], model["sel_size"]), resample=SAMPLING_MODE,
            )
        )
        out[:, :, -1] = 255
        out_pil = Image.fromarray(out)
        out_buffer = io.BytesIO()
        out_pil.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        base64_bytes = base64.b64encode(out_buffer.read())
        base64_str = base64_bytes.decode("ascii")
    return (
        gr.update(label=str(state + 1), value=base64_str,),
        gr.update(label="Prompt"),
        state + 1,
    )


def load_js(name):
    if name in ["export", "commit", "undo"]:
        return f"""
function (x)
{{ 
    let frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
    let button=frame.querySelector("#{name}");
    button.click();
    return x;
}}
"""
    ret = ""
    with open(f"./js/{name}.js", "r") as f:
        ret = f.read()
    return ret


upload_button_js = load_js("upload")
outpaint_button_js = load_js("outpaint")
proceed_button_js = load_js("proceed")
mode_js = load_js("mode")
setup_button_js = load_js("setup")
with blocks as demo:
    # title
    title = gr.Markdown(
        """
    **stablediffusion-infinity**: Outpainting with Stable Diffusion on an infinite canvas: [https://github.com/lkwq007/stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity)
    """
    )
    # frame
    frame = gr.HTML(test(2), visible=False)
    # setup
    with gr.Row():
        token = gr.Textbox(
            label="Huggingface token",
            value=get_token(),
            placeholder="Input your token here",
        )
        canvas_width = gr.Number(
            label="Canvas width", value=1024, precision=0, elem_id="canvas_width"
        )
        canvas_height = gr.Number(
            label="Canvas height", value=600, precision=0, elem_id="canvas_height"
        )
        selection_size = gr.Number(
            label="Selection box size", value=384, precision=0, elem_id="selection_size"
        )
    setup_button = gr.Button("Setup (may take a while)", variant="primary")
    with gr.Row():
        with gr.Column(scale=3, min_width=270):
            # canvas control
            canvas_control = gr.Radio(
                label="Control",
                choices=[PAINT_SELECTION, IMAGE_SELECTION, BRUSH_SELECTION],
                value=PAINT_SELECTION,
                elem_id="control",
            )
            with gr.Box():
                with gr.Group():
                    run_button = gr.Button(value="Outpaint")
                    export_button = gr.Button(value="Export")
                    commit_button = gr.Button(value="✓")
                    retry_button = gr.Button(value="⟳")
                    undo_button = gr.Button(value="↶")
        with gr.Column(scale=3, min_width=270):
            sd_prompt = gr.Textbox(
                label="Prompt", placeholder="input your prompt here", lines=4
            )
        with gr.Column(scale=2, min_width=150):
            with gr.Box():
                sd_resize = gr.Checkbox(label="Resize input to 515x512", value=True)
                safety_check = gr.Checkbox(label="Enable Safety Checker", value=True)
            sd_strength = gr.Slider(
                label="Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.01
            )
        with gr.Column(scale=1, min_width=150):
            sd_step = gr.Number(label="Step", value=50, precision=0)
            sd_guidance = gr.Number(label="Guidance", value=7.5)
    with gr.Row():
        with gr.Column(scale=4, min_width=600):
            init_mode = gr.Radio(
                label="Init mode",
                choices=[
                    "patchmatch",
                    "edge_pad",
                    "cv2_ns",
                    "cv2_telea",
                    "gaussian",
                    "perlin",
                ],
                value="patchmatch",
                type="value",
            )

    proceed_button = gr.Button("Proceed", elem_id="proceed", visible=DEBUG_MODE)
    # sd pipeline parameters
    with gr.Accordion("Upload image", open=False):
        image_box = gr.Image(image_mode="RGBA", source="upload", type="pil")
        upload_button = gr.Button(
            "Before uploading the image you need to setup the canvas first"
        )
    model_output = gr.Textbox(visible=DEBUG_MODE, elem_id="output", label="0")
    model_input = gr.Textbox(visible=DEBUG_MODE, elem_id="input", label="Input")
    upload_output = gr.Textbox(visible=DEBUG_MODE, elem_id="upload", label="0")
    model_output_state = gr.State(value=0)
    upload_output_state = gr.State(value=0)
    # canvas_state = gr.State({"width":1024,"height":600,"selection_size":384})

    def upload_func(image, state):
        pil = image.convert("RGBA")
        w, h = pil.size
        if w > model["width"] - 100 or h > model["height"] - 100:
            pil = contain_func(pil, (model["width"] - 100, model["height"] - 100))
        out_buffer = io.BytesIO()
        pil.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        base64_bytes = base64.b64encode(out_buffer.read())
        base64_str = base64_bytes.decode("ascii")
        return (
            gr.update(label=str(state + 1), value=base64_str),
            state + 1,
        )

    upload_button.click(
        fn=upload_func,
        inputs=[image_box, upload_output_state],
        outputs=[upload_output, upload_output_state],
        _js=upload_button_js,
    )

    def setup_func(token_val, width, height, size):
        model["width"] = width
        model["height"] = height
        model["sel_size"] = size
        try:
            get_model(token_val)
        except Exception as e:
            return {token: gr.update(value="Invalid token!")}
        return {
            token: gr.update(visible=False),
            canvas_width: gr.update(visible=False),
            canvas_height: gr.update(visible=False),
            selection_size: gr.update(visible=False),
            setup_button: gr.update(visible=False),
            frame: gr.update(visible=True),
            upload_button: gr.update(value="Upload"),
        }

    setup_button.click(
        fn=setup_func,
        inputs=[token, canvas_width, canvas_height, selection_size],
        outputs=[
            token,
            canvas_width,
            canvas_height,
            selection_size,
            setup_button,
            frame,
            upload_button,
        ],
        _js=setup_button_js,
    )
    run_button.click(
        fn=None, inputs=[run_button], outputs=[run_button], _js=outpaint_button_js,
    )
    retry_button.click(
        fn=None, inputs=[run_button], outputs=[run_button], _js=outpaint_button_js,
    )
    proceed_button.click(
        fn=run_outpaint,
        inputs=[
            model_input,
            sd_prompt,
            sd_strength,
            sd_guidance,
            sd_step,
            sd_resize,
            init_mode,
            safety_check,
            model_output_state,
        ],
        outputs=[model_output, sd_prompt, model_output_state],
        _js=proceed_button_js,
    )
    export_button.click(
        fn=None, inputs=[export_button], outputs=[export_button], _js=load_js("export")
    )
    commit_button.click(
        fn=None, inputs=[export_button], outputs=[export_button], _js=load_js("commit")
    )
    undo_button.click(
        fn=None, inputs=[export_button], outputs=[export_button], _js=load_js("undo")
    )
    canvas_control.change(
        fn=None, inputs=[canvas_control], outputs=[canvas_control], _js=mode_js,
    )
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="stablediffusion-infinity")
    parser.add_argument("--port", type=int, help="listen port", default=7860)
    parser.add_argument("--host", type=str, help="host", default="127.0.0.1")
    parser.add_argument(
        "--share", type=bool, action="store_true", help="share this app?"
    )
    args = parser.parse_args()
    if args.share:
        demo.launch(share=True)
    else:
        demo.launch(server_name=args.host, server_port=args.port)

