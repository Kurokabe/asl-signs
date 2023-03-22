FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda11.7.1

WORKDIR /ASL
COPY . .
RUN pip install pyarrow