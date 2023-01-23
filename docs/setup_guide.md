# Setup Guide

Please install conda at first ([miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://docs.anaconda.com/anaconda/install/)). 

- [Setup with Linux/Nvidia GPU](#linux)
- [Setup with Linux/AMD GPU](#linux-amd)
- [Setup with Windows](#windows-nvidia)
- [Setup with MacOS](#macos)
- [Upgrade from previous version](#upgrade)

## Setup with Linux/Nvidia GPU <a name="linux"></a>

### conda env
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
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy accelerate
pip install opencv-python
pip install -U gradio
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3
pip install timm
```

After setup the environment, you can run stablediffusion-infinity with following commands:
```
conda activate sd-inf
python app.py
```

## Setup with Linux/AMD GPU <a name="linux-amd"></a> (untested)

```
conda create -n sd-inf python=3.10
conda activate sd-inf
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy accelerate
pip install opencv-python
pip install -U gradio
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3
pip install timm
```


### CPP library (optional)

Note that `opencv` library (e.g. `libopencv-dev`/`opencv-devel`, the package name may differ on different distributions) is required for `PyPatchMatch`. You may need to install `opencv` by yourself. If no `opencv` installed, the `patch_match` option (usually better quality) won't work. 

## Setup with Windows <a name="windows"></a>


```
conda create -n sd-inf python=3.10
conda activate sd-inf
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy accelerate
pip install opencv-python
pip install -U gradio
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3
pip install timm
```

If you use AMD GPUs, you need to install the ONNX runtime `pip install onnxruntime-directml` (only works with the `stablediffusion-inpainting` model, untested on AMD devices). 

For windows, you may need to replace `pip install opencv-python` with `conda install -c conda-forge opencv`

After setup the environment, you can run stablediffusion-infinity with following commands:
```
conda activate sd-inf
python app.py
```
## Setup with MacOS <a name="macos"></a>

### conda env
```
conda create -n sd-inf python=3.10
conda activate sd-inf
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy accelerate
pip install opencv-python
pip install -U gradio
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3
pip install timm
```

After setup the environment, you can run stablediffusion-infinity with following commands:
```
conda activate sd-inf
python app.py
```
### CPP library (optional)

Note that `opencv` library is required for `PyPatchMatch`. You may need to install `opencv` by yourself (via `homebrew` or compile from source). If no `opencv` installed, the `patch_match` option (usually better quality) won't work. 

## Upgrade <a name="upgrade"></a>

```
conda install -c conda-forge diffusers transformers ftfy accelerate
conda update -c conda-forge diffusers transformers ftfy accelerate
pip install -U gradio
```