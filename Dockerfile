# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for lightgbm and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with strict versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . .

# Install the package in development mode
RUN pip install -e .

# Download model files
RUN python -m alma_classifier.download_models

# Set Numba environment variables to avoid caching issues
ENV NUMBA_CACHE_DIR=/tmp
ENV NUMBA_DISABLE_PERFORMANCE_WARNINGS=1

# Create volume mount point for data
VOLUME ["/data"]

# Set the default command
ENTRYPOINT ["alma-classifier"]
CMD ["--help"]