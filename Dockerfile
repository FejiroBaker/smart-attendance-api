FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install opencv-python-headless==4.8.0.76

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT