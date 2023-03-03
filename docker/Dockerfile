FROM continuumio/miniconda3:4.12.0 AS build

RUN git clone https://github.com/dhecloud/st_AED_deliverables.git
WORKDIR st_AED_deliverables/

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "streaming", "/bin/bash", "-c"]
RUN conda update ffmpeg -y
RUN conda install  -n streaming pytorch torchvision=0.13 torchaudio cpuonly -c pytorch 

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streaming", "python", "api.py"]