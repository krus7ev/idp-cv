# Use Python 3.13 slim image for a balance of size and compatibility with spacy
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required for OpenCV and potential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
# Note: We use --extra-index-url for CPU-only torch wheels as defined in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the local package (this will automatically trigger model downloads via setup.py)
RUN pip install .

# Provide a default entrypoint indicating how to run the script
ENTRYPOINT ["python", "run_extraction.py"]
CMD ["-h"]
