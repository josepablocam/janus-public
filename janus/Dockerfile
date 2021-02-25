FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get install -y build-essential
RUN apt-get install -y task-spooler
RUN apt-get install -y vim
RUN apt-get install -y apt-utils
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y zip
RUN apt-get install -y git
RUN apt-get install -y htop
RUN apt-get -y install gcc

COPY . janus/
COPY  datasets datasets/
COPY  kaggle kaggle/

WORKDIR janus/
RUN rm -rf datasets/
RUN rm -rf kaggle/

RUN bash scripts/install.sh

ENTRYPOINT bash
