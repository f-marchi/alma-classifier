############################
# Base image
############################

FROM python:3.11-slim AS runtime

ARG ALMA_VERSION=0.2.0

WORKDIR /app

############################
# System dependencies
############################
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    procps \
    bash \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    && rm -rf /var/lib/apt/lists/*

############################
# Dependency installation (leverage layer caching)
############################
COPY pyproject.toml README.md requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source AFTER dependency install for better caching
COPY alma_classifier ./alma_classifier

# Install package (production, non-editable) to embed version metadata
RUN pip install --no-cache-dir .

############################
# Download model artifacts at build time (can be skipped with --build-arg SKIP_MODEL_DOWNLOAD=1)
############################
ARG SKIP_MODEL_DOWNLOAD=0
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ]; then python -m alma_classifier.download_models; else echo "Skipping model download"; fi

############################
# Runtime environment variables
############################
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp \
    NUMBA_DISABLE_PERFORMANCE_WARNINGS=1

############################
# Non-root user
############################
RUN useradd -m -u 1000 almauser && chown -R almauser:almauser /app
USER almauser

############################
# Smoke test (help message)
############################
RUN alma-classifier --help

############################
# Default command
############################
CMD ["alma-classifier", "--help"]