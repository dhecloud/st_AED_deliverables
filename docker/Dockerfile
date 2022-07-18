FROM continuumio/miniconda3 AS build

RUN git clone https://github.com/dhecloud/st_AED_deliverables.git
WORKDIR st_AED_deliverables/


RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "streaming", "/bin/bash", "-c"]
RUN conda install  -n streaming pytorch torchvision cpuonly -c pytorch 
RUN conda install -c conda-forge conda-pack
RUN conda-pack -n streaming -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

FROM debian:buster AS runtime
COPY --from=build /venv /venv

SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/dhecloud/st_AED_deliverables.git
WORKDIR st_AED_deliverables/
ENTRYPOINT source /venv/bin/activate && \
           python api.py
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streaming", "python", "api.py"]