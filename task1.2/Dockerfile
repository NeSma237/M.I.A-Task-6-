# Base image: slim Python — smaller image size, doesn't include unnecessary
# build tools we don't need for CPU-only inference.
FROM python:3.11-slim

WORKDIR /app

# System dependency needed by Pillow/torchvision for image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (before the rest of the code) so Docker can cache
# this layer — dependencies rarely change between builds, but code does.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project: source code, the deployment app, and the
# trained model checkpoint + vocabulary.
COPY src/ ./src/
COPY app/ ./app/
COPY checkpoints/ ./checkpoints/

# Gradio's default port — matches server_port=7860 in gradio_app.py.
EXPOSE 7860

# Runs gradio_app.py as a module, exactly like we tested locally.
CMD ["python", "-m", "app.gradio_app"]