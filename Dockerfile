FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

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

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy project files
COPY pyproject.toml uv.lock* LICENSE README.md ./
COPY src/ ./src/

# Install the project and dependencies
RUN uv sync --frozen --no-dev

# Install pip in the uv environment for spaCy model downloads
RUN uv pip install pip

# Download spaCy models
RUN uv run python -m spacy download en_core_web_sm
RUN uv run python -m spacy download zh_core_web_sm

# Download sentence-transformers model
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Set the entrypoint
ENTRYPOINT ["uv", "run", "bilingual-align"]