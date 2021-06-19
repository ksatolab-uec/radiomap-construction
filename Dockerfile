FROM python:3.8.10-slim
MAINTAINER k_sato

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools \
    && pip install --no-cache-dir \
    numpy==1.20 \
    matplotlib==3.4.2 \
    scipy==1.6.3