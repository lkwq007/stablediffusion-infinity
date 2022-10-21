# Usage

## Models

- stablediffusion-inpainting: `runwayml/stable-diffusion-inpainting`, does not support img2img mode
- stablediffusion-inpainting+img2img-v1.5: `runwayml/stable-diffusion-inpainting` + `runwayml/stable-diffusion-v1-5`, supports img2img mode, requires larger vRAM
- stablediffusion-v1.5: `runwayml/stable-diffusion-v1-5`, inpainting with `diffusers`'s legacy pipeline, low quality for outpainting, supports img2img mode
- stablediffusion-v1.4: `CompVis/stable-diffusion-v1-4`, inpainting with `diffusers`'s legacy pipeline, low quality for outpainting, supports img2img mode

## Loading local model

Note that when loading a local checkpoint, you have to specify the correct model choice before setup. 
```shell
python app.py --local_model path_to_local_model
# e.g. 
# diffusers model weights
python app.py --local_model ./models/runwayml/stable-diffusion-inpainting
python app.py --local_model models/CompVis/stable-diffusion-v1-4/model_index.json
# original model checkpoint
python app.py --local_model /home/user/checkpoint/model.ckpt
```

## Loading remote model

Note that when loading a remote model, you have to specify the correct model choice before setup. 
```shell
python app.py --remote_model model_name
# e.g. 
python app.py --remote_model hakurei/waifu-diffusion-v1-3
```

## Using textual inversion embeddings 

Put `*.bin` inside `embeddings` directory. 

## Using a dreambooth finetuned model

```
python app.py --remote_model model_name
# e.g.
python app.py --remote_model sd-dreambooth-library/pikachu
# or download the weight/checkpoint and load with
python app.py --local_model path_to_model
```

## Model Path for Docker users

Docker users can specify a local model path or remote mode name within the web app. 

## Using fp32 mode or low vRAM mode (some GPUs might not work well fp16)

```shell
python app.py --fp32 --lowvram
```

## HTTPS

```shell
python app.py --encrypt --ssl_keyfile path_to_ssl_keyfile --ssl_certfile path_to_ssl_certfile
```

## Keyboard shortcut

The shortcut can be configured via `config.yaml`. Currently only support `[key]` or `[Ctrl]` + `[key]`

Default shortcuts are: 

```yaml
shortcut:
  clear: Escape
  load: Ctrl+o
  save: Ctrl+s
  export: Ctrl+e
  upload: Ctrl+u
  selection: 1
  canvas: 2
  eraser: 3
  outpaint: d
  accept: a
  cancel: c
  retry: r
  prev: q
  next: e
  zoom_in: z
  zoom_out: x
  random_seed: s
```

## Glossary

(From diffusers' document https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
- prompt: The prompt to guide the image generation.
- step: The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. 
- guidance_scale: Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,usually at the expense of lower image quality.
- negative_prompt: The prompt or prompts not to guide the image generation.
- Sample number: The number of images to generate per prompt
- scheduler: A scheduler is used in combination with `unet` to denoise the encoded image latens.
- eta: Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to DDIMScheduler, will be ignored for others.
- strength: for img2img only, Conceptually, indicates how much to transform the reference image.