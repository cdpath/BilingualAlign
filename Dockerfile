FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
      sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
      sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
      sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download zh_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the application code
COPY book_aligner.py .

# Create directories for input and output files
RUN mkdir -p /app/books /app/output

# Set the entrypoint
ENTRYPOINT ["python", "book_aligner.py"]