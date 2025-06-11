# Bilingual Book Sentence Aligner

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings and evaluates translation quality using LLMs.


## Tools

### 1. Sentence Alignment

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings.

#### Usage

##### with UV

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

##### with Docker

```bash
docker build -t bilingual-aligner .
docker run -v $(pwd)/sample_books:/app/books -v $(pwd)/output:/app/output bilingual-aligner books/english_sample.md books/chinese_sample.md output/aligned_output.tsv
```

#### Format Requirements

##### Input Format

Markdown files with H1 and H2 headings for chapter segmentation.

##### Output Format

- **TSV**: English Sentence, Chinese Sentence, Alignment Score
- **JSON**: List of objects with english, chinese, and score fields

### 2. Translation Quality Evaluation

Evaluates translation quality using LLMs with detailed error analysis and scoring across multiple dimensions.


#### Usage

```bash
# Evaluate aligned sentences using Mistral Medium (default: 3 sentence pairs per call)
export MISTRAL_API_KEY="your_api_key_here"
uv run src/evaluate.py input.tsv output.tsv

# Use different model with custom settings
uv run src/evaluate.py input.tsv output.tsv --model gpt-4 --temperature 0.2

# Process more sentence pairs per LLM call for better token efficiency
uv run src/evaluate.py input.tsv output.tsv --eval-batch-size 5

# Generate summary report
uv run src/evaluate.py input.tsv output.tsv --report summary.json

# Use custom prompt
uv run src/evaluate.py input.tsv output.tsv --prompt custom_prompt.txt

# Enable verbose logging
uv run src/evaluate.py input.tsv output.tsv --verbose
```

#### Configuration Options

- `--model, -m`: LLM model to use (default: mistral/mistral-medium)
- `--eval-batch-size, -e`: Number of sentence pairs per LLM call (default: 3)
- `--temperature, -t`: Temperature for LLM (default: 0.3)
- `--max-tokens`: Maximum tokens for LLM response (default: 2000)
- `--batch-size, -b`: Batch size for rate limiting (default: 5)
- `--timeout`: API timeout in seconds (default: 30)
- `--report, -r`: Output JSON file for summary report
- `--prompt, -p`: Custom prompt file
- `--verbose, -v`: Enable verbose logging

## Example Workflow

1. **Align sentences**: 
   ```bash
   uv run bilingual-align english_book.md chinese_book.md aligned_sentences.tsv
   ```

2. **Evaluate translation quality**:
   ```bash
   export MISTRAL_API_KEY="your_api_key"
   uv run src/evaluate.py aligned_sentences.tsv evaluation_results.tsv --report summary.json
   ```

3. **Review results**: Check detailed evaluations in `evaluation_results.tsv` and summary statistics in `summary.json`
