FROM continuumio/miniconda3:4.12.0

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN apt-get update && \
    apt install -y \
      fonts-dejavu-core  \
      build-essential \
      libopencv-dev \
      cmake \
      vim \
      && apt-get clean

COPY opencv.pc /usr/lib/pkgconfig/opencv.pc

RUN useradd -ms /bin/bash user && mkdir /app && chown user:user /app
USER user
RUN mkdir ~/.huggingface && conda init bash
WORKDIR /app
EXPOSE 8888