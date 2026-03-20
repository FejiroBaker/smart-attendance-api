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

# Pre-download the InsightFace buffalo_sc model (~85 MB) at build time.
# Using || true so a warmup failure doesn't break the build —
# the model will download on first request instead.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace model cached successfully')" || echo "Warmup skipped - model will load at runtime"

# Railway sets $PORT dynamically
ENV PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT