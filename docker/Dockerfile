FROM continuumio/miniconda3:4.12.0

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN apt-get update && \
    apt install -y \
      fonts-dejavu-core  \
      build-essential \
      libopencv-dev \
      && apt-get clean

COPY opencv.pc /usr/lib/pkgconfig/opencv.pc

RUN useradd -ms /bin/bash user
USER user
RUN mkdir ~/.huggingface && mkdir ~/app && conda init bash
WORKDIR /home/user/app
EXPOSE 8888