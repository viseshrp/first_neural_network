FROM continuumio/miniconda3
ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

COPY env.yaml /code
COPY . /code
RUN conda env create -f env.yaml
