#!/bin/bash

cd /app

set -euxo pipefail

set -x

git submodule update --init --recursive
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
