# stablediffusion-infinity

Outpainting with Stable Diffusion on an infinite canvas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lkwq007/stablediffusion-infinity/blob/master/stablediffusion_infinity_colab.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/lnyan/stablediffusion-infinity)

Start with init_image (updated demo in Gradio):




https://user-images.githubusercontent.com/1665437/193394123-d202efc8-24a7-41b3-a5cf-6b2e0b60db28.mp4



Start with text2img ([ipycanvas](https://github.com/lkwq007/stablediffusion-infinity/tree/ipycanvas) version):

https://user-images.githubusercontent.com/1665437/190212025-f4a82c46-0ff1-4ca2-b79b-6c81601e3eed.mp4


The web app might work on Windows (see this issue https://github.com/lkwq007/stablediffusion-infinity/issues/12 for more information) and Apple Silicon devices (untested, check guide here: https://huggingface.co/docs/diffusers/optimization/mps). 

## Status

This project mainly works as a proof of concept. In that case, ~~the UI design is relatively weak~~, and the quality of results is not guaranteed. 
You may need to do prompt engineering or change the size of the selection box to get better outpainting results. 

The project now becomes a web app based on PyScript and Gradio. For Jupyter Notebook version, please check out the [ipycanvas](https://github.com/lkwq007/stablediffusion-infinity/tree/ipycanvas) branch. 

Pull requests are welcome for better UI control, ideas to achieve better results, or any other improvements.

Update: the project also supports [glid-3-xl-stable](https://github.com/Jack000/glid-3-xl-stable) as inpainting/outpainting model. Note that you have to restart the `app.py` to change model. (not supported on colab)

## Setup environment
setup with `environment.yml`
```
git clone --recurse-submodules https://github.com/lkwq007/stablediffusion-infinity
cd stablediffusion-infinity
conda env create -f environment.yml
```

if the `environment.yml` doesn't work for you, you may install dependencies manually: 
```
conda create -n sd-inf python=3.10
conda activate sd-inf
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy
pip install opencv-python
pip install gradio==3.4.0
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3

```

For windows, you may need to replace `pip install opencv-python` with `conda install -c conda-forge opencv`
## CPP library (optional)

Note that `opencv` library (e.g. `libopencv-dev`/`opencv-devel`, the package name may differ on different distributions) is required for `PyPatchMatch`. You may need to install `opencv` by yourself. If no `opencv` installed, the `patch_match` option (usually better quality) won't work. 

## How-to

```
conda activate sd-inf
python app.py
```

## Running with Docker

This should get you started without needing to manually install anything, except for having an environment with Docker installed and an Nvidia GPU.
This has been tested on Docker Desktop on Windows 10 using the WSL2 backend.

First, update the .env file with your Huggingface token from https://huggingface.co/settings/tokens

Open your shell that has docker and run these commands

```
cd stablediffusion-infinity
docker-compose build
docker-compose up
```
Open "http://localhost:8888" in your browser ( even though the log says http://0.0.0.0:8888 )

## FAQs

- Troubleshooting on Windows: 
  - https://github.com/lkwq007/stablediffusion-infinity/issues/12
- False positive rate of safety checker is quite high: 
  - https://github.com/lkwq007/stablediffusion-infinity/issues/8#issuecomment-1248448453
- What is the init_mode
  - init_mode indicates how to fill the empty/masked region, usually `patch_match` is better than others
- Why not use `postMessage` for iframe interaction
  - The iframe the gradio are in the same origin. For `postMessage` version, check out [gradio-space](https://github.com/lkwq007/stablediffusion-infinity/tree/gradio-space)

## Credit

The code of `perlin2d.py` is from https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy/42154921#42154921 and is **not** included in the scope of LICENSE used in this repo.

The submodule `glid_3_xl_stable` is based on https://github.com/Jack000/glid-3-xl-stable 

The submodule `PyPatchMatch` is based on https://github.com/vacancy/PyPatchMatch
