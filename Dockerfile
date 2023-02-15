FROM continuumio/miniconda3:4.12.0 AS build

#RUN git clone -b 23feb8_resnet https://github.com/dhecloud/st_AED_deliverables.git

WORKDIR /app
COPY ./ ./

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "streaming", "/bin/bash", "-c"]
RUN conda update ffmpeg -y
RUN conda install  -n streaming pytorch torchvision=0.13 torchaudio cpuonly -c pytorch 

#RUN conda install -c conda-forge conda-pack
#RUN conda-pack --ignore-missing-files -n streaming -o /tmp/env.tar && \
#  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#  rm /tmp/env.tar

#RUN /venv/bin/conda-unpack

#FROM debian:buster AS runtime
#COPY --from=build /venv /venv

#SHELL ["/bin/bash", "-c"]
#RUN apt-get update
#RUN apt-get install ffmpeg git -y

#RUN git clone -b 23feb8_resnet https://github.com/dhecloud/st_AED_deliverables.git
#WORKDIR st_AED_deliverables/
#WORKDIR /app
#COPY ./ ./

#ENTRYPOINT source /venv/bin/activate && \
#           python api.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streaming", "python", "api.py"]