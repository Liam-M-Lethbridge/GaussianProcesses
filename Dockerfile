FROM python:3.12-slim

# Basic Python hygiene
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Container working directory
WORKDIR /app

# Install system deps (safe default)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy source code
COPY . .

# Make project importable
ENV PYTHONPATH=/app

# Start in shell (interactive)
CMD ["bash"]
