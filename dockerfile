# Use the official Python base image
FROM python:3.10.14-bookworm

# Set working directory
WORKDIR /app

# Install all dependencies directly via pip
RUN pip install --no-cache-dir \
    dash==2.18.2 \
    dash-bootstrap-components \
    numpy==2.0.2 \
    pandas==2.2.3 \
    scikit-learn==1.6.0 \
    mlflow==2.20.2 \
    matplotlib==3.10.0 \
    psutil==6.1.1 \
    scipy==1.15.0 \
    pytest==8.3.5 \
    cloudpickle==3.1.0\
    defusedxml==0.7.1

COPY . /app


CMD tail -f /dev/null