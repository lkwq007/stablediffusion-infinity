# stablediffusion-infinity

Outpainting with Stable Diffusion on an infinite canvas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lkwq007/stablediffusion-infinity/blob/master/stablediffusion_infinity_colab.ipynb)

Start with init_image:



https://user-images.githubusercontent.com/1665437/190231611-fc263115-0fb9-4f2d-a71b-7e500c1e311d.mp4


Start with text2img:

https://user-images.githubusercontent.com/1665437/190212025-f4a82c46-0ff1-4ca2-b79b-6c81601e3eed.mp4


It is recommended to run the notebook on a local server for better interactive control. 

The notebook might work on Windows (see this issue https://github.com/lkwq007/stablediffusion-infinity/issues/12 for more information) and Apple Silicon devices (untested, check guide here: https://huggingface.co/docs/diffusers/optimization/mps). 

## Status

This project mainly works as a proof of concept. In that case, the UI design is relatively weak, and the quality of results is not guaranteed. 
You may need to do prompt engineering or change the size of the selection box to get better outpainting results. 

Pull requests are welcome for better UI control, ideas to achieve better results, or any other improvements. 

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
pip install gradio==3.4
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

Watch the log for the url to open in your browser. Choose the one that starts with http://127.0.0.1:8888/


## FAQs

- Troubleshooting on Windows: 
  - https://github.com/lkwq007/stablediffusion-infinity/issues/12
- False positive rate of safety checker is quite high: 
  - https://github.com/lkwq007/stablediffusion-infinity/issues/8#issuecomment-1248448453
- What is the init_mode
  - init_mode indicates how to fill the empty/masked region, usually `patch_match` is better than others
- The GUI is lagging on colab
  - It is recommended to run the notebook on a local server since the interactions and canvas content updates are actually handled by the python backend on the serverside, and that's how `ipycanvas` works
  - colab doesn't support the latest version of `ipycanvas`, which may have better performance

## Credit

The code of `perlin2d.py` is from https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy/42154921#42154921 and is **not** included in the scope of LICENSE used in this repo.
