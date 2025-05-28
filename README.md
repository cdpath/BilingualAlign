# Bilingual Book Sentence Aligner

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings.

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

## Input Format

Markdown files with H1 and H2 headings for chapter segmentation.

## Output Format

TSV: English Sentence, Chinese Sentence, Alignment Score
JSON: List of objects with english, chinese, and score fields

## Limitations

- Currently aligns only first chapters
- Alignment quality depends on embedding model and similarity threshold (0.5)
