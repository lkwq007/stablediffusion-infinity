# stablediffusion-infinity

Outpainting with Stable Diffusion on an infinite canvas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lkwq007/stablediffusion-infinity/blob/master/stablediffusion_infinity_colab.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/lnyan/stablediffusion-infinity)
[![Setup Locally](https://img.shields.io/badge/%F0%9F%96%A5%EF%B8%8F%20Setup-Locally-blue)](https://github.com/lkwq007/stablediffusion-infinity/blob/master/docs/setup_guide.md)

Start with init_image (updated demo in Gradio):

https://user-images.githubusercontent.com/1665437/193394123-d202efc8-24a7-41b3-a5cf-6b2e0b60db28.mp4


## Status

This project mainly works as a proof of concept. In that case, ~~the UI design is relatively weak~~, and the quality of results is not guaranteed. 
You may need to do prompt engineering or change the size of the selection box to get better outpainting results. 

The project now becomes a web app based on PyScript and Gradio. For Jupyter Notebook version, please check out the [ipycanvas](https://github.com/lkwq007/stablediffusion-infinity/tree/ipycanvas) branch. 

Pull requests are welcome for better UI control, ideas to achieve better results, or any other improvements.

Update: the project also supports [glid-3-xl-stable](https://github.com/Jack000/glid-3-xl-stable) as inpainting/outpainting model. Note that you have to restart the `app.py` to change model. (not supported on colab)

Update: the project add photometric correction to suppress seams, to use this feature, you need to install [fpie](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing): `pip install fpie` (Linux/MacOS only)

## Docs

### Get Started

- Setup for Windows: [setup_guide](./docs/setup_guide.md#windows)
- Setup for Linux: [setup_guide](./docs/setup_guide.md#linux)
- Setup for MacOS: [setup_guide](./docs/setup_guide.md#macos)
- Running with Docker on Windows or Linux with NVIDIA GPU: [run_with_docker](./docs/run_with_docker.md)

### FAQs

- The result is a black square: 
  - False positive rate of safety checker is relatively high, you may disable the safety_checker
- What is the init_mode
  - init_mode indicates how to fill the empty/masked region, usually `patch_match` is better than others
- Why not use `postMessage` for iframe interaction
  - The iframe and the gradio are in the same origin. For `postMessage` version, check out [gradio-space](https://github.com/lkwq007/stablediffusion-infinity/tree/gradio-space) version

## Credit

The code of `perlin2d.py` is from https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy/42154921#42154921 and is **not** included in the scope of LICENSE used in this repo.

The submodule `glid_3_xl_stable` is based on https://github.com/Jack000/glid-3-xl-stable 

The submodule `PyPatchMatch` is based on https://github.com/vacancy/PyPatchMatch

The code of `postprocess.py` and `process.py` is modified based on https://github.com/Trinkle23897/Fast-Poisson-Image-Editing

The code of `convert_checkpoint.py` is modified based on https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
