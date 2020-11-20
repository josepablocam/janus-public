FROM ubuntu:18.04

RUN apt-get update
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

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID janus-user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID janus-user
WORKDIR /home/janus-user

COPY --chown=janus-user . janus/
COPY --chown=janus-user datasets datasets/
COPY --chown=janus-user kaggle kaggle/

WORKDIR janus/
RUN rm -rf datasets/
RUN rm -rf kaggle/

USER janus-user
RUN bash scripts/install.sh --kaggle

ENTRYPOINT bash
