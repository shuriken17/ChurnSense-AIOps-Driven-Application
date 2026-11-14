# Shared base image: Python 3.12 plus every library of requirements.lock.
# The trainer, the model server and the tests all start from this image, so serving uses
# exactly the library versions that trained the model.
# After the first successful build, pin the base by digest (docker image inspect python:3.12-slim).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src

# libgomp1: LightGBM needs it (python:3.12-slim does not ship it). git: DVC needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock
