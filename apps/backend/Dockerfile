FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/tmp/torch_cache \
    XDG_CACHE_HOME=/tmp \
    HF_HOME=/tmp/hf \
    HF_HUB_CACHE=/tmp/hf

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps (requirements.txt is at repo root in the Space)
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip wheel && \
    pip install -r requirements.txt && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2

# App code (the whole backend subtree becomes the Space root)
COPY . /app

# Expose Space port
ENV PORT=7860
EXPOSE 7860

# Start FastAPI
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]