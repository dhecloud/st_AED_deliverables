FROM continuumio/miniconda3
RUN git clone https://github.com/dhecloud/st_AED_deliverables.git
RUN cd st_AED_deliverables
RUN conda env create -f ./environment.yml
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda activate streaming
CMD python api.py