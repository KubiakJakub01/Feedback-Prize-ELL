FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /pipeline

COPY requirements.txt /pipeline
# Install dependencies
RUN pip install -r /pipeline/requirements.txt

# Copy the rest of the code
COPY src /pipeline
