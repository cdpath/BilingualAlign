# Bilingual Book Sentence Aligner

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings.

## Usage

### with UV

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install the project and its dependencies (including spaCy models):

```bash
uv sync
```

3. Run

```bash
# uv run bilingual-align <english_book.md> <chinese_book.md> <output_file> [--format tsv|json]
uv run bilingual-align sample_books/english_sample.md sample_books/chinese_sample.md output/aligned_output.tsv
```


### with Docker


```bash
docker build -t bilingual-aligner .
docker run -v $(pwd)/sample_books:/app/books -v $(pwd)/output:/app/output bilingual-aligner books/english_sample.md books/chinese_sample.md output/aligned_output.tsv
```


## Format Requirements

### Input Format

Markdown files with H1 and H2 headings for chapter segmentation.

### Output Format

- **TSV**: English Sentence, Chinese Sentence, Alignment Score
- **JSON**: List of objects with english, chinese, and score fields

