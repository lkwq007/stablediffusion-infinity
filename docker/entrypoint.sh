#!/bin/bash

cd ~/app

if [ "$HUGGINGFACE_TOKEN" = "" ] ||  [ "$HUGGINGFACE_TOKEN" = "your_token_here" ] ; then
  echo "Missing HUGGINGFACE_TOKEN, visit https://huggingface.co/settings/tokens and put it into the stablediffusion-infinity/.env file."
  echo "or set and pass the HUGGINGFACE_TOKEN environment variable."
  exit 1
fi

echo -n "$HUGGINGFACE_TOKEN" > /home/user/.huggingface/token

set -x

git submodule init
git submodule update

if ! conda env list | grep sd-inf ; then
    git config --global credential.helper store
    echo "Creating environment, wait a few minutes..."
    conda env create -f environment.yml
    echo "conda activate sd-inf" >> ~/.bashrc
fi

. "/opt/conda/etc/profile.d/conda.sh"
conda activate sd-inf

jupyter lab --ip=0.0.0.0 --port=8888
