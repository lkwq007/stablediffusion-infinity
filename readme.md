# stablediffusion-infinity

Outpainting with Stable Diffusion on an infinite canvas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lkwq007/stablediffusion-infinity/blob/master/stablediffusion_infinity_colab.ipynb)

Start with init_image:



https://user-images.githubusercontent.com/1665437/190231611-fc263115-0fb9-4f2d-a71b-7e500c1e311d.mp4


Start with text2img:

https://user-images.githubusercontent.com/1665437/190212025-f4a82c46-0ff1-4ca2-b79b-6c81601e3eed.mp4


It is recommended to run the notebook on a local server for better interactive control. 

The notebook might work on Windows (see this issue https://github.com/lkwq007/stablediffusion-infinity/issues/12 for more information) and Apple Silicon devices (untested, check guide here: https://huggingface.co/docs/diffusers/optimization/mps). 

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
conda install scipy
conda install -c conda-forge jupyterlab
conda install -c conda-forge ipywidgets=7.7.1
conda install -c conda-forge ipycanvas
conda install -c conda-forge diffusers transformers ftfy
pip install opencv-python
```

For windows, you may need to replace `pip install opencv-python` with `conda install -c conda-forge opencv`
## CPP library (optional)

Note that `opencv` library (e.g. `libopencv-dev`/`opencv-devel`, the package name may differ on different distributions) is required for `PyPatchMatch`. You may need to install `opencv` by yourself. If no `opencv` installed, the `patch_match` option (usually better quality) won't work. 

## How-to

```
conda activate sd-inf
huggingface-cli login # ignore this if you have already logged in
jupyter lab
# and then open stablediffusion_infinity.ipynb and run cells

```
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

