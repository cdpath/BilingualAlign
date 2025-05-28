# Bilingual Book Sentence Aligner

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings.

## Requirements

- Python 3.9+
- PyTorch 2.3.0+ (required for sentence-transformers compatibility)
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download spaCy models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

## Usage

```bash
python book_aligner.py <english_book.md> <chinese_book.md> <output_file> [--format tsv|json]
```

Example:

```bash
python book_aligner.py english_sample.md chinese_sample.md aligned_output.tsv
```

## Docker Usage

Build and run with Docker:

```bash
docker build -t bilingual-aligner .
docker run -v $(pwd)/sample_books:/app/books -v $(pwd)/output:/app/output bilingual-aligner books/english_sample.md books/chinese_sample.md output/aligned_output.tsv
```

## Input Format

Markdown files with H1 and H2 headings for chapter segmentation.

## Output Format

TSV: English Sentence, Chinese Sentence, Alignment Score
JSON: List of objects with english, chinese, and score fields

## Troubleshooting

If you encounter the error `module 'torch' has no attribute 'get_default_device'`, ensure you have PyTorch 2.3.0 or higher installed. This function was introduced in PyTorch 2.3.0 and is required by the latest sentence-transformers library.

## Limitations

- Currently aligns only first chapters
- Alignment quality depends on embedding model and similarity threshold (0.5)
