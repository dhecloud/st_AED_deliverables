FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update 
RUN apt-get install wget git -y

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN git clone https://github.com/dhecloud/st_AED_deliverables.git
WORKDIR st_AED_deliverables/

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "streaming", "/bin/bash", "-c"]
RUN conda install  -n streaming pytorch torchvision torchaudio cpuonly -c pytorch
RUN python -c "import torch"


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streaming", "python", "api.py"]