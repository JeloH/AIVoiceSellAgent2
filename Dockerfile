# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY scripts/api/requirements.txt /app/scripts/api/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /app/scripts/api/requirements.txt

COPY . /app
RUN chmod +x /app/scripts/api/docker-entrypoint.sh

ENV HOST=0.0.0.0
ENV PORT=5000
ENV LORA_PATH=/app/scripts/model/abc_energy_lora_adapter2
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["/app/scripts/api/docker-entrypoint.sh"]