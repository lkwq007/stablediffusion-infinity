# stablediffusion-infinity


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lkwq007/stablediffusion-infinity/blob/master/stablediffusion_infinity_colab.ipynb)

Start with init_image:



https://user-images.githubusercontent.com/1665437/190231611-fc263115-0fb9-4f2d-a71b-7e500c1e311d.mp4


Start with text2img:

https://user-images.githubusercontent.com/1665437/190212025-f4a82c46-0ff1-4ca2-b79b-6c81601e3eed.mp4





Outpainting with Stable Diffusion on an infinite canvas

It is recommended run the notebook on a local server. 
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
conda install -c conda-forge ipycanvas
conda install -c conda-forge diffusers transformers ftfy
pip install opencv-python
```

Note that `opencv` library is required for `PyPatchMatch`. You may need to install `opencv` by yourself. 

## How-to

```
conda activate sd-inf
huggingface-cli login # ignore this if you have already logged in
jupyter lab
# and then open stablediffusion_infinity.ipynb and run cells

```
