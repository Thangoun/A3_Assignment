FROM python:3.11.4-bookworm

WORKDIR /app

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn==1.6.0
RUN pip3 install mlflow


COPY . /app


CMD tail -f /dev/null