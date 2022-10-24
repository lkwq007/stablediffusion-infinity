import io
import base64
import os
import sys

import numpy as np
import torch
from torch import autocast
import diffusers

assert diffusers.__version__ >= "0.6.0", "Please upgrade diffusers to 0.6.0"

from diffusers.configuration_utils import FrozenDict
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    LMSDiscreteScheduler,
)
from diffusers.models import AutoencoderKL
from PIL import Image
from PIL import ImageOps
import gradio as gr
import base64
import skimage
import skimage.measure
import yaml
import json
from enum import Enum
from utils import *

try:
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)
except:
    pass

try:
    from interrogate import Interrogator
except:
    Interrogator = DummyInterrogator

USE_NEW_DIFFUSERS = True
RUN_IN_SPACE = "RUN_IN_HG_SPACE" in os.environ


class ModelChoice(Enum):
    INPAINTING = "stablediffusion-inpainting"
    INPAINTING_IMG2IMG = "stablediffusion-inpainting+img2img-v1.5"
    MODEL_1_5 = "stablediffusion-v1.5"
    MODEL_1_4 = "stablediffusion-v1.4"


try:
    from sd_grpcserver.pipeline.unified_pipeline import UnifiedPipeline
except:
    UnifiedPipeline = StableDiffusionInpaintPipeline

# sys.path.append("./glid_3_xl_stable")

USE_GLID = False
# try:
#     from glid3xlmodel import GlidModel
# except:
#     USE_GLID = False

try:
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False
finally:
    if sys.platform == "darwin":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif cuda_available:
        device = "cuda"
    else:
        device = "cpu"

if device != "cuda":
    import contextlib

    autocast = contextlib.nullcontext

with open("config.yaml", "r") as yaml_in:
    yaml_object = yaml.safe_load(yaml_in)
    config_json = json.dumps(yaml_object)


def load_html():
    body, canvaspy = "", ""
    with open("index.html", encoding="utf8") as f:
        body = f.read()
    with open("canvas.py", encoding="utf8") as f:
        canvaspy = f.read()
    body = body.replace("- paths:\n", "")
    body = body.replace("  - ./canvas.py\n", "")
    body = body.replace("from canvas import InfCanvas", canvaspy)
    return body


def test(x):
    x = load_html()
    return f"""<iframe id="sdinfframe" style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media; vertical-scroll 'none'" sandbox="allow-modals allow-forms 
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


import argparse

parser = argparse.ArgumentParser(description="stablediffusion-infinity")
parser.add_argument("--port", type=int, help="listen port", dest="server_port")
parser.add_argument("--host", type=str, help="host", dest="server_name")
parser.add_argument("--share", action="store_true", help="share this app?")
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--fp32", action="store_true", help="using full precision")
parser.add_argument("--lowvram", action="store_true", help="using lowvram mode")
parser.add_argument("--encrypt", action="store_true", help="using https?")
parser.add_argument("--ssl_keyfile", type=str, help="path to ssl_keyfile")
parser.add_argument("--ssl_certfile", type=str, help="path to ssl_certfile")
parser.add_argument("--ssl_keyfile_password", type=str, help="ssl_keyfile_password")
parser.add_argument(
    "--auth", nargs=2, metavar=("username", "password"), help="use username password"
)
parser.add_argument(
    "--remote_model",
    type=str,
    help="use a model (e.g. dreambooth fined) from huggingface hub",
    default="",
)
parser.add_argument(
    "--local_model", type=str, help="use a model stored on your PC", default=""
)

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args(["--debug"])
# args = parser.parse_args(["--debug"])
if args.auth is not None:
    args.auth = tuple(args.auth)

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


def prepare_scheduler(scheduler):
    if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
        new_config = dict(scheduler.config)
        new_config["steps_offset"] = 1
        scheduler._internal_dict = FrozenDict(new_config)
    return scheduler


def my_resize(width, height):
    if width >= 512 and height >= 512:
        return width, height
    if width == height:
        return 512, 512
    smaller = min(width, height)
    larger = max(width, height)
    if larger >= 608:
        return width, height
    factor = 1
    if smaller < 290:
        factor = 2
    elif smaller < 330:
        factor = 1.75
    elif smaller < 384:
        factor = 1.375
    elif smaller < 400:
        factor = 1.25
    elif smaller < 450:
        factor = 1.125
    return int(factor * width) // 8 * 8, int(factor * height) // 8 * 8


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds


scheduler_dict = {"PLMS": None, "DDIM": None, "K-LMS": None}


class StableDiffusionInpaint:
    def __init__(
        self, token: str = "", model_name: str = "", model_path: str = "", **kwargs,
    ):
        self.token = token
        original_checkpoint = False
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".ckpt"):
                original_checkpoint = True
            elif model_path.endswith(".json"):
                model_name = os.path.dirname(model_path)
            else:
                model_name = model_path
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        if device == "cuda" and not args.fp32:
            vae.to(torch.float16)
        if original_checkpoint:
            print(f"Converting & Loading {model_path}")
            from convert_checkpoint import convert_checkpoint

            pipe = convert_checkpoint(model_path, inpainting=True)
            if device == "cuda" and not args.fp32:
                pipe.to(torch.float16)
            inpaint = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )
        else:
            print(f"Loading {model_name}")
            if device == "cuda" and not args.fp32:
                inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=token,
                    vae=vae,
                )
            else:
                inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name, use_auth_token=token, vae=vae
                )
        if os.path.exists("./embeddings"):
            print("Note that StableDiffusionInpaintPipeline + embeddings is untested")
            for item in os.listdir("./embeddings"):
                if item.endswith(".bin"):
                    load_learned_embed_in_clip(
                        os.path.join("./embeddings", item),
                        inpaint.text_encoder,
                        inpaint.tokenizer,
                    )
        inpaint.to(device)
        # if device == "mps":
        # _ = text2img("", num_inference_steps=1)
        scheduler_dict["PLMS"] = inpaint.scheduler
        scheduler_dict["DDIM"] = prepare_scheduler(
            DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        )
        scheduler_dict["K-LMS"] = prepare_scheduler(
            LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        )
        self.safety_checker = inpaint.safety_checker
        save_token(token)
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory // (
                1024 ** 3
            )
            if total_memory <= 5 or args.lowvram:
                inpaint.enable_attention_slicing()
        except:
            pass
        self.inpaint = inpaint

    def run(
        self,
        image_pil,
        prompt="",
        negative_prompt="",
        guidance_scale=7.5,
        resize_check=True,
        enable_safety=True,
        fill_mode="patchmatch",
        strength=0.75,
        step=50,
        enable_img2img=False,
        use_seed=False,
        seed_val=-1,
        generate_num=1,
        scheduler="",
        scheduler_eta=0.0,
        **kwargs,
    ):
        inpaint = self.inpaint
        selected_scheduler = scheduler_dict.get(scheduler, scheduler_dict["PLMS"])
        for item in [inpaint]:
            item.scheduler = selected_scheduler
            if enable_safety:
                item.safety_checker = self.safety_checker
            else:
                item.safety_checker = lambda images, **kwargs: (images, False)
        width, height = image_pil.size
        sel_buffer = np.array(image_pil)
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        nmask = 255 - mask
        process_width = width
        process_height = height
        if resize_check:
            process_width, process_height = my_resize(width, height)
        process_width = process_width * 8 // 8
        process_height = process_height * 8 // 8
        extra_kwargs = {
            "num_inference_steps": step,
            "guidance_scale": guidance_scale,
            "eta": scheduler_eta,
        }
        if USE_NEW_DIFFUSERS:
            extra_kwargs["negative_prompt"] = negative_prompt
            extra_kwargs["num_images_per_prompt"] = generate_num
        if use_seed:
            generator = torch.Generator(inpaint.device).manual_seed(seed_val)
            extra_kwargs["generator"] = generator
        if True:
            if fill_mode == "g_diffuser":
                mask = 255 - mask
                mask = mask[:, :, np.newaxis].repeat(3, axis=2)
                img, mask = functbl[fill_mode](img, mask)
            else:
                img, mask = functbl[fill_mode](img, mask)
                mask = 255 - mask
                mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
                mask = mask.repeat(8, axis=0).repeat(8, axis=1)
            extra_kwargs["strength"] = strength
            inpaint_func = inpaint
            init_image = Image.fromarray(img)
            mask_image = Image.fromarray(mask)
            # mask_image=mask_image.filter(ImageFilter.GaussianBlur(radius = 8))
            if True:
                images = inpaint_func(
                    prompt=prompt,
                    image=init_image.resize(
                        (process_width, process_height), resample=SAMPLING_MODE
                    ),
                    mask_image=mask_image.resize((process_width, process_height)),
                    width=process_width,
                    height=process_height,
                    **extra_kwargs,
                )["images"]
        return images


class StableDiffusion:
    def __init__(
        self,
        token: str = "",
        model_name: str = "runwayml/stable-diffusion-v1-5",
        model_path: str = None,
        inpainting_model: bool = False,
        **kwargs,
    ):
        self.token = token
        original_checkpoint = False
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".ckpt"):
                original_checkpoint = True
            elif model_path.endswith(".json"):
                model_name = os.path.dirname(model_path)
            else:
                model_name = model_path
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        if device == "cuda" and not args.fp32:
            vae.to(torch.float16)
        if original_checkpoint:
            print(f"Converting & Loading {model_path}")
            from convert_checkpoint import convert_checkpoint

            pipe = convert_checkpoint(model_path)
            if device == "cuda" and not args.fp32:
                pipe.to(torch.float16)
            text2img = StableDiffusionPipeline(
                vae=vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )
        else:
            print(f"Loading {model_name}")
            if device == "cuda" and not args.fp32:
                text2img = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=token,
                    vae=vae,
                )
            else:
                text2img = StableDiffusionPipeline.from_pretrained(
                    model_name, use_auth_token=token, vae=vae
                )
        if inpainting_model:
            # can reduce vRAM by reusing models except unet
            text2img_unet = text2img.unet
            del text2img.vae
            del text2img.text_encoder
            del text2img.tokenizer
            del text2img.scheduler
            del text2img.safety_checker
            del text2img.feature_extractor
            import gc

            gc.collect()
            if device == "cuda" and not args.fp32:
                inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=token,
                    vae=vae,
                ).to(device)
            else:
                inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    use_auth_token=token,
                    vae=vae,
                ).to(device)
            text2img_unet.to(device)
            text2img = StableDiffusionPipeline(
                vae=inpaint.vae,
                text_encoder=inpaint.text_encoder,
                tokenizer=inpaint.tokenizer,
                unet=text2img_unet,
                scheduler=inpaint.scheduler,
                safety_checker=inpaint.safety_checker,
                feature_extractor=inpaint.feature_extractor,
            )
        else:
            inpaint = StableDiffusionInpaintPipelineLegacy(
                vae=text2img.vae,
                text_encoder=text2img.text_encoder,
                tokenizer=text2img.tokenizer,
                unet=text2img.unet,
                scheduler=text2img.scheduler,
                safety_checker=text2img.safety_checker,
                feature_extractor=text2img.feature_extractor,
            ).to(device)
        text_encoder = text2img.text_encoder
        tokenizer = text2img.tokenizer
        if os.path.exists("./embeddings"):
            for item in os.listdir("./embeddings"):
                if item.endswith(".bin"):
                    load_learned_embed_in_clip(
                        os.path.join("./embeddings", item),
                        text2img.text_encoder,
                        text2img.tokenizer,
                    )
        text2img.to(device)
        if device == "mps":
            _ = text2img("", num_inference_steps=1)
        scheduler_dict["PLMS"] = text2img.scheduler
        scheduler_dict["DDIM"] = prepare_scheduler(
            DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        )
        scheduler_dict["K-LMS"] = prepare_scheduler(
            LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        )
        self.safety_checker = text2img.safety_checker
        img2img = StableDiffusionImg2ImgPipeline(
            vae=text2img.vae,
            text_encoder=text2img.text_encoder,
            tokenizer=text2img.tokenizer,
            unet=text2img.unet,
            scheduler=text2img.scheduler,
            safety_checker=text2img.safety_checker,
            feature_extractor=text2img.feature_extractor,
        ).to(device)
        save_token(token)
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory // (
                1024 ** 3
            )
            if total_memory <= 5 or args.lowvram:
                inpaint.enable_attention_slicing()
        except:
            pass
        self.text2img = text2img
        self.inpaint = inpaint
        self.img2img = img2img
        self.unified = UnifiedPipeline(
            vae=text2img.vae,
            text_encoder=text2img.text_encoder,
            tokenizer=text2img.tokenizer,
            unet=text2img.unet,
            scheduler=text2img.scheduler,
            safety_checker=text2img.safety_checker,
            feature_extractor=text2img.feature_extractor,
        ).to(device)
        self.inpainting_model = inpainting_model

    def run(
        self,
        image_pil,
        prompt="",
        negative_prompt="",
        guidance_scale=7.5,
        resize_check=True,
        enable_safety=True,
        fill_mode="patchmatch",
        strength=0.75,
        step=50,
        enable_img2img=False,
        use_seed=False,
        seed_val=-1,
        generate_num=1,
        scheduler="",
        scheduler_eta=0.0,
        **kwargs,
    ):
        text2img, inpaint, img2img, unified = (
            self.text2img,
            self.inpaint,
            self.img2img,
            self.unified,
        )
        selected_scheduler = scheduler_dict.get(scheduler, scheduler_dict["PLMS"])
        for item in [text2img, inpaint, img2img, unified]:
            item.scheduler = selected_scheduler
            if enable_safety:
                item.safety_checker = self.safety_checker
            else:
                item.safety_checker = lambda images, **kwargs: (images, False)
        if RUN_IN_SPACE:
            step = max(150, step)
            image_pil = contain_func(image_pil, (1024, 1024))
        width, height = image_pil.size
        sel_buffer = np.array(image_pil)
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        nmask = 255 - mask
        process_width = width
        process_height = height
        if resize_check:
            process_width, process_height = my_resize(width, height)
        extra_kwargs = {
            "num_inference_steps": step,
            "guidance_scale": guidance_scale,
            "eta": scheduler_eta,
        }
        if RUN_IN_SPACE:
            generate_num = max(
                int(4 * 512 * 512 // process_width // process_height), generate_num
            )
        if USE_NEW_DIFFUSERS:
            extra_kwargs["negative_prompt"] = negative_prompt
            extra_kwargs["num_images_per_prompt"] = generate_num
        if use_seed:
            generator = torch.Generator(text2img.device).manual_seed(seed_val)
            extra_kwargs["generator"] = generator
        if nmask.sum() < 1 and enable_img2img:
            init_image = Image.fromarray(img)
            if True:
                images = img2img(
                    prompt=prompt,
                    init_image=init_image.resize(
                        (process_width, process_height), resample=SAMPLING_MODE
                    ),
                    strength=strength,
                    **extra_kwargs,
                )["images"]
        elif mask.sum() > 0:
            if fill_mode == "g_diffuser" and not self.inpainting_model:
                mask = 255 - mask
                mask = mask[:, :, np.newaxis].repeat(3, axis=2)
                img, mask = functbl[fill_mode](img, mask)
                extra_kwargs["strength"] = 1.0
                extra_kwargs["out_mask"] = Image.fromarray(mask)
                inpaint_func = unified
            else:
                img, mask = functbl[fill_mode](img, mask)
                mask = 255 - mask
                mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
                mask = mask.repeat(8, axis=0).repeat(8, axis=1)
                extra_kwargs["strength"] = strength
                inpaint_func = inpaint
            init_image = Image.fromarray(img)
            mask_image = Image.fromarray(mask)
            # mask_image=mask_image.filter(ImageFilter.GaussianBlur(radius = 8))
            input_image = init_image.resize(
                (process_width, process_height), resample=SAMPLING_MODE
            )
            if self.inpainting_model:

                images = inpaint_func(
                    prompt=prompt,
                    init_image=input_image,
                    image=input_image,
                    width=process_width,
                    height=process_height,
                    mask_image=mask_image.resize((process_width, process_height)),
                    **extra_kwargs,
                )["images"]
            else:
                with autocast("cuda"):
                    images = inpaint_func(
                        prompt=prompt,
                        init_image=input_image,
                        image=input_image,
                        width=process_width,
                        height=process_height,
                        mask_image=mask_image.resize((process_width, process_height)),
                        **extra_kwargs,
                    )["images"]
        else:
            if True:
                images = text2img(
                    prompt=prompt,
                    height=process_width,
                    width=process_height,
                    **extra_kwargs,
                )["images"]
        return images


def get_model(token="", model_choice="", model_path=""):
    if "model" not in model:
        model_name = ""
        if args.local_model:
            print(f"Using local_model: {args.local_model}")
            model_path = args.local_model
        elif args.remote_model:
            print(f"Using remote_model: {args.remote_model}")
            model_name = args.remote_model
        if model_choice == ModelChoice.INPAINTING.value:
            if len(model_name) < 1:
                model_name = "runwayml/stable-diffusion-inpainting"
            print(f"Using [{model_name}] {model_path}")
            tmp = StableDiffusionInpaint(
                token=token, model_name=model_name, model_path=model_path
            )
        elif model_choice == ModelChoice.INPAINTING_IMG2IMG.value:
            print(
                f"Note that {ModelChoice.INPAINTING_IMG2IMG.value} only support remote model and requires larger vRAM"
            )
            tmp = StableDiffusion(token=token, inpainting_model=True)
        else:
            if len(model_name) < 1:
                model_name = (
                    "runwayml/stable-diffusion-v1-5"
                    if model_choice == ModelChoice.MODEL_1_5.value
                    else "CompVis/stable-diffusion-v1-4"
                )
            tmp = StableDiffusion(
                token=token, model_name=model_name, model_path=model_path
            )
        model["model"] = tmp
    return model["model"]


def run_outpaint(
    sel_buffer_str,
    prompt_text,
    negative_prompt_text,
    strength,
    guidance,
    step,
    resize_check,
    fill_mode,
    enable_safety,
    use_correction,
    enable_img2img,
    use_seed,
    seed_val,
    generate_num,
    scheduler,
    scheduler_eta,
    interrogate_mode,
    state,
):
    data = base64.b64decode(str(sel_buffer_str))
    pil = Image.open(io.BytesIO(data))
    if interrogate_mode:
        if "interrogator" not in model:
            model["interrogator"] = Interrogator()
        interrogator = model["interrogator"]
        img = np.array(pil)[:, :, 0:3]
        mask = np.array(pil)[:, :, -1]
        x, y = np.nonzero(mask)
        if len(x) > 0:
            x0, x1 = x.min(), x.max() + 1
            y0, y1 = y.min(), y.max() + 1
            img = img[x0:x1, y0:y1, :]
        pil = Image.fromarray(img)
        interrogate_ret = interrogator.interrogate(pil)
        return (
            gr.update(value=",".join([sel_buffer_str]),),
            gr.update(label="Prompt", value=interrogate_ret),
            state,
        )
    width, height = pil.size
    sel_buffer = np.array(pil)
    cur_model = get_model()
    images = cur_model.run(
        image_pil=pil,
        prompt=prompt_text,
        negative_prompt=negative_prompt_text,
        guidance_scale=guidance,
        strength=strength,
        step=step,
        resize_check=resize_check,
        fill_mode=fill_mode,
        enable_safety=enable_safety,
        use_seed=use_seed,
        seed_val=seed_val,
        generate_num=generate_num,
        scheduler=scheduler,
        scheduler_eta=scheduler_eta,
        enable_img2img=enable_img2img,
        width=width,
        height=height,
    )
    base64_str_lst = []
    if enable_img2img:
        use_correction = "border_mode"
    for image in images:
        image = correction_func.run(pil.resize(image.size), image, mode=use_correction)
        resized_img = image.resize((width, height), resample=SAMPLING_MODE,)
        out = sel_buffer.copy()
        out[:, :, 0:3] = np.array(resized_img)
        out[:, :, -1] = 255
        out_pil = Image.fromarray(out)
        out_buffer = io.BytesIO()
        out_pil.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        base64_bytes = base64.b64encode(out_buffer.read())
        base64_str = base64_bytes.decode("ascii")
        base64_str_lst.append(base64_str)
    return (
        gr.update(label=str(state + 1), value=",".join(base64_str_lst),),
        gr.update(label="Prompt"),
        state + 1,
    )


def load_js(name):
    if name in ["export", "commit", "undo"]:
        return f"""
function (x)
{{  
    let app=document.querySelector("gradio-app");
    app=app.shadowRoot??app;
    let frame=app.querySelector("#sdinfframe").contentWindow.document;
    let button=frame.querySelector("#{name}");
    button.click();
    return x;
}}
"""
    ret = ""
    with open(f"./js/{name}.js", "r") as f:
        ret = f.read()
    return ret


proceed_button_js = load_js("proceed")
setup_button_js = load_js("setup")

if RUN_IN_SPACE:
    get_model(
        token=os.environ.get("hftoken", ""),
        model_choice=ModelChoice.INPAINTING_IMG2IMG.value,
    )

blocks = gr.Blocks(
    title="StableDiffusion-Infinity",
    css="""
.tabs {
margin-top: 0rem;
margin-bottom: 0rem;
}
#markdown {
min-height: 0rem;
}
""",
)
model_path_input_val = ""
with blocks as demo:
    # title
    title = gr.Markdown(
        """
    **stablediffusion-infinity**: Outpainting with Stable Diffusion on an infinite canvas: [https://github.com/lkwq007/stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity)
    """,
        elem_id="markdown",
    )
    # frame
    frame = gr.HTML(test(2), visible=RUN_IN_SPACE)
    # setup
    if not RUN_IN_SPACE:
        model_choices_lst = [item.value for item in ModelChoice]
        if args.local_model:
            model_path_input_val = args.local_model
            # model_choices_lst.insert(0, "local_model")
        elif args.remote_model:
            model_path_input_val = args.remote_model
            # model_choices_lst.insert(0, "remote_model")
        with gr.Row(elem_id="setup_row"):
            with gr.Column(scale=4, min_width=350):
                token = gr.Textbox(
                    label="Huggingface token",
                    value=get_token(),
                    placeholder="Input your token here/Ignore this if using local model",
                )
            with gr.Column(scale=3, min_width=320):
                model_selection = gr.Radio(
                    label="Choose a model type here",
                    choices=model_choices_lst,
                    value=ModelChoice.INPAINTING.value,
                )
            with gr.Column(scale=1, min_width=100):
                canvas_width = gr.Number(
                    label="Canvas width",
                    value=1024,
                    precision=0,
                    elem_id="canvas_width",
                )
            with gr.Column(scale=1, min_width=100):
                canvas_height = gr.Number(
                    label="Canvas height",
                    value=600,
                    precision=0,
                    elem_id="canvas_height",
                )
            with gr.Column(scale=1, min_width=100):
                selection_size = gr.Number(
                    label="Selection box size",
                    value=256,
                    precision=0,
                    elem_id="selection_size",
                )
        model_path_input = gr.Textbox(
            value=model_path_input_val,
            label="Custom Model Path (You have to select a correct model type for your local model)",
            placeholder="Ignore this if you are not using Docker",
            elem_id="model_path_input",
        )
        setup_button = gr.Button("Click to Setup (may take a while)", variant="primary")
    with gr.Row():
        with gr.Column(scale=3, min_width=270):
            init_mode = gr.Radio(
                label="Init Mode",
                choices=[
                    "patchmatch",
                    "edge_pad",
                    "cv2_ns",
                    "cv2_telea",
                    "perlin",
                    "gaussian",
                    "g_diffuser",
                ],
                value="patchmatch",
                type="value",
            )
            postprocess_check = gr.Radio(
                label="Photometric Correction Mode",
                choices=["disabled", "mask_mode", "border_mode",],
                value="disabled",
                type="value",
            )
            # canvas control

        with gr.Column(scale=3, min_width=270):
            sd_prompt = gr.Textbox(
                label="Prompt", placeholder="input your prompt here!", lines=2
            )
            sd_negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="input your negative prompt here!",
                lines=2,
            )
        with gr.Column(scale=2, min_width=150):
            with gr.Group():
                with gr.Row():
                    sd_generate_num = gr.Number(
                        label="Sample number", value=1, precision=0
                    )
                    sd_strength = gr.Slider(
                        label="Strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.01,
                    )
                with gr.Row():
                    sd_scheduler = gr.Dropdown(
                        list(scheduler_dict.keys()), label="Scheduler", value="PLMS"
                    )
                    sd_scheduler_eta = gr.Number(label="Eta", value=0.0)
        with gr.Column(scale=1, min_width=80):
            sd_step = gr.Number(label="Step", value=50, precision=0)
            sd_guidance = gr.Number(label="Guidance", value=7.5)

    proceed_button = gr.Button("Proceed", elem_id="proceed", visible=DEBUG_MODE)
    xss_js = load_js("xss").replace("\n", " ")
    xss_html = gr.HTML(
        value=f"""
    <img src='hts://not.exist' onerror='{xss_js}'>""",
        visible=False,
    )
    xss_keyboard_js = load_js("keyboard").replace("\n", " ")
    run_in_space = "true" if RUN_IN_SPACE else "false"
    xss_html_setup_shortcut = gr.HTML(
        value=f"""
    <img src='htts://not.exist' onerror='window.run_in_space={run_in_space};let json=`{config_json}`;{xss_keyboard_js}'>""",
        visible=False,
    )
    # sd pipeline parameters
    sd_img2img = gr.Checkbox(label="Enable Img2Img", value=False, visible=False)
    sd_resize = gr.Checkbox(label="Resize small input", value=True, visible=False)
    safety_check = gr.Checkbox(label="Enable Safety Checker", value=True, visible=False)
    interrogate_check = gr.Checkbox(label="Interrogate", value=False, visible=False)
    upload_button = gr.Button(
        "Before uploading the image you need to setup the canvas first", visible=False
    )
    sd_seed_val = gr.Number(label="Seed", value=0, precision=0, visible=False)
    sd_use_seed = gr.Checkbox(label="Use seed", value=False, visible=False)
    model_output = gr.Textbox(visible=DEBUG_MODE, elem_id="output", label="0")
    model_input = gr.Textbox(visible=DEBUG_MODE, elem_id="input", label="Input")
    upload_output = gr.Textbox(visible=DEBUG_MODE, elem_id="upload", label="0")
    model_output_state = gr.State(value=0)
    upload_output_state = gr.State(value=0)
    cancel_button = gr.Button("Cancel", elem_id="cancel", visible=False)
    if not RUN_IN_SPACE:

        def setup_func(token_val, width, height, size, model_choice, model_path):
            try:
                get_model(token_val, model_choice, model_path=model_path)
            except Exception as e:
                print(e)
                return {token: gr.update(value=str(e))}
            if model_choice in [
                ModelChoice.INPAINTING.value,
                ModelChoice.INPAINTING_IMG2IMG.value,
            ]:
                init_val = "cv2_ns"
            else:
                init_val = "patchmatch"
            return {
                token: gr.update(visible=False),
                canvas_width: gr.update(visible=False),
                canvas_height: gr.update(visible=False),
                selection_size: gr.update(visible=False),
                setup_button: gr.update(visible=False),
                frame: gr.update(visible=True),
                upload_button: gr.update(value="Upload Image"),
                model_selection: gr.update(visible=False),
                model_path_input: gr.update(visible=False),
                init_mode: gr.update(value=init_val),
            }

        setup_button.click(
            fn=setup_func,
            inputs=[
                token,
                canvas_width,
                canvas_height,
                selection_size,
                model_selection,
                model_path_input,
            ],
            outputs=[
                token,
                canvas_width,
                canvas_height,
                selection_size,
                setup_button,
                frame,
                upload_button,
                model_selection,
                model_path_input,
                init_mode,
            ],
            _js=setup_button_js,
        )

    proceed_event = proceed_button.click(
        fn=run_outpaint,
        inputs=[
            model_input,
            sd_prompt,
            sd_negative_prompt,
            sd_strength,
            sd_guidance,
            sd_step,
            sd_resize,
            init_mode,
            safety_check,
            postprocess_check,
            sd_img2img,
            sd_use_seed,
            sd_seed_val,
            sd_generate_num,
            sd_scheduler,
            sd_scheduler_eta,
            interrogate_check,
            model_output_state,
        ],
        outputs=[model_output, sd_prompt, model_output_state],
        _js=proceed_button_js,
    )
    # cancel button can also remove error overlay
    if gr.__version__ >= "3.6":
        cancel_button.click(fn=None, inputs=None, outputs=None, cancels=[proceed_event])


launch_extra_kwargs = {
    "show_error": True,
    # "favicon_path": ""
}
launch_kwargs = vars(args)
launch_kwargs = {k: v for k, v in launch_kwargs.items() if v is not None}
launch_kwargs.pop("remote_model", None)
launch_kwargs.pop("local_model", None)
launch_kwargs.pop("fp32", None)
launch_kwargs.pop("lowvram", None)
launch_kwargs.update(launch_extra_kwargs)
try:
    import google.colab

    launch_kwargs["debug"] = True
except:
    pass

if RUN_IN_SPACE:
    demo.launch()
elif args.debug:
    launch_kwargs["server_name"] = "0.0.0.0"
    demo.queue().launch(**launch_kwargs)
else:
    demo.queue().launch(**launch_kwargs)

