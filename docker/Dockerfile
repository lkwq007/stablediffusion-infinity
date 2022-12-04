FROM continuumio/miniconda3:4.12.0

RUN apt-get update && \
    apt install -y \
      fonts-dejavu-core  \
      build-essential \
      libopencv-dev \
      cmake \
      vim \
      && apt-get clean

COPY docker/opencv.pc /usr/lib/pkgconfig/opencv.pc

RUN useradd -ms /bin/bash user
USER user

RUN mkdir ~/.huggingface && conda init bash

COPY --chown=user:user . /app
WORKDIR /app

EXPOSE 8888
CMD ["/app/docker/entrypoint.sh"]