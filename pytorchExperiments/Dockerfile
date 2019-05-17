from pytorch/pytorch
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y python3-pip
RUN conda install pandas
RUN mkdir /workspace/data
COPY ./data/* /workspace/data/
COPY *.py /workspace/
