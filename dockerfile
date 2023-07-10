FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /pipeline

COPY requirements.txt /pipeline
RUN pip install -r /pipeline/requirements.txt

COPY src /pipeline
