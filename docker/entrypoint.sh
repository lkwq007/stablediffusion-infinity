#!/bin/bash

set -euxo pipefail
cd /app

if [ "$HUGGINGFACE_TOKEN" = "" ] ||  [ "$HUGGINGFACE_TOKEN" = "your_token_here" ] ; then
  echo "Missing HUGGINGFACE_TOKEN, visit https://huggingface.co/settings/tokens and put it into the stablediffusion-infinity/.env file."
  echo "or set and pass the HUGGINGFACE_TOKEN environment variable."
  exit 1
fi

echo -n "$HUGGINGFACE_TOKEN" > /home/user/.huggingface/token

set -x

git submodule init
git submodule update
git config --global credential.helper store
if ! conda env list | grep sd-inf ; then
    echo "Creating environment, it may appear to freeze for a few minutes..."
    conda env create -f environment.yml
    echo "Finished installing."
    echo "conda activate sd-inf" >> ~/.bashrc
    shasum environment.yml > ~/.environment.sha
fi

. "/opt/conda/etc/profile.d/conda.sh"
conda activate sd-inf

if shasum -c ~/.environment.sha > /dev/null 2>&1 ; then
  echo "environment.yml is unchanged."
else
  echo "environment.yml was changed, please wait a minute until it says 'Done updating'..."
  conda env update --file environment.yml
  shasum environment.yml > ~/.environment.sha
  echo "Done updating."
fi

python app.py --port=8888 --host=0.0.0.0
