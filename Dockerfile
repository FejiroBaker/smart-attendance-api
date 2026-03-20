FROM python:3.11-slim

# System libs: GL for opencv + gcc/g++ to compile insightface's C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-download the InsightFace buffalo_sc model at build time
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace model cached successfully')" || echo "Warmup skipped - model will load at runtime"

# Use shell form (not exec form) so $PORT variable expands correctly at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}