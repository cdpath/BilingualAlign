# Bilingual Book Sentence Aligner

## Overview
This Python tool aligns English and Chinese sentences from books provided in Markdown format. It segments books into chapters, allows for chapter alignment (currently simplified), segments chapters into sentences, and then uses sentence embeddings to find corresponding sentences across the two languages.

## Features
*   Chapter segmentation from Markdown files using H1 and H2 headings.
*   Chapter alignment (Note: Current version uses a placeholder for user interaction, aligning the first chapter of each book by default).
*   Sentence segmentation using spaCy models for English and Chinese, with NLTK/regex fallbacks.
*   Sentence alignment using state-of-the-art multilingual sentence embedding models (`sentence-transformers`).
*   Output of aligned sentences in TSV or JSON formats.

## Prerequisites
*   Python 3.7+
*   pip

## Installation
1.  Clone the repository (if applicable) or download the `book_aligner.py` script and `requirements.txt`.
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download spaCy models required for sentence segmentation:
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download zh_core_web_sm
    ```

## Usage
Basic command structure:
```bash
python book_aligner.py <path_to_english_book.md> <path_to_chinese_book.md> <output_file_path> [--format <format>]
```

Arguments:
*   `english_book_path`: Path to the English Markdown book.
*   `chinese_book_path`: Path to the Chinese Markdown book.
*   `output_file_path`: Desired path for the alignment output file.
*   `--format <format>`: Optional. Specifies the output format. Can be `tsv` (default) or `json`.

Example:
```bash
python book_aligner.py ./sample_books/english_sample.md ./sample_books/chinese_sample.md ./output/aligned_output.tsv --format tsv
```

## Input File Format
*   Input books must be in Markdown (`.md`) format.
*   The tool uses H1 (`#`) and H2 (`##`) headings to identify and segment chapters.

## Output File Format
*   **TSV**: A tab-separated values file with columns: `English Sentence`, `Chinese Sentence`, `Alignment Score`.
*   **JSON**: A JSON file containing a list of objects. Each object has keys: `english` (the English sentence), `chinese` (the aligned Chinese sentence), and `score` (the alignment similarity score).

## Current Limitations & Future Work
*   **Chapter Alignment:** The current version aligns only the first chapter of the English book with the first chapter of the Chinese book as a demonstration. A future version will include interactive prompts for users to map chapters accurately.
*   **Accuracy:** The quality of sentence alignment depends on the chosen embedding model, the similarity threshold (currently hardcoded at 0.5), and the quality/nature of the input texts.
*   **Error Handling:** Basic error handling is in place, but can be expanded.
