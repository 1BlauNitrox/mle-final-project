# Miniconda 25.3.1-1 is the published Python 3.13 image. A fixed base prevents
# the course Dockerfile from silently moving to an unsupported Python runtime.
FROM continuumio/miniconda3:25.3.1-1
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD /bin/bash
