from tensorflow/tensorflow:1.4.0-gpu-py3
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:neovim-ppa/stable -y
RUN apt-get update
RUN apt-get install neovim -y
RUN mkdir /notebooks/recurrent-relational-networks
COPY ./recurrent-relational-networks /notebooks/recurrent-relational-networks/
RUN echo 'export PYTHONPATH="{PYTHONPATH}:/notebooks/recurrent-relational-networks/"'>> ~/.bashrc
