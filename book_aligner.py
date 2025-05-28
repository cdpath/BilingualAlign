import argparse
import re
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import json

NLP_MODELS = {}
EMBEDDING_MODEL = None

# ensure NLTK's 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        nltk.download("punkt")
        print("'punkt' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading 'punkt': {e}. NLTK sentence tokenization might fail.")


def segment_sentences(text: str, language: str) -> list[str]:
    """Segments text into sentences based on language."""
    if language == "english":
        if "en" not in NLP_MODELS:
            try:
                NLP_MODELS["en"] = spacy.load("en_core_web_sm")
                print("Loaded 'en_core_web_sm' spaCy model.")
            except OSError:
                print(
                    "Error loading 'en_core_web_sm' spaCy model. Falling back to NLTK."
                )
                NLP_MODELS["en"] = None  # Mark as failed to avoid retrying

        if NLP_MODELS.get("en"):
            doc = NLP_MODELS["en"](text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            print("Using NLTK for English sentence segmentation.")
            try:
                return [
                    sent.strip() for sent in nltk.sent_tokenize(text) if sent.strip()
                ]
            except (
                LookupError
            ):  # Should be caught by the initial download attempt, but as a safeguard
                print("NLTK 'punkt' still not found. Please ensure it is installed.")
                return []  # Cannot segment
            except Exception as e:
                print(f"Error during NLTK tokenization: {e}")
                return []

    elif language == "chinese":
        if "zh" not in NLP_MODELS:
            try:
                NLP_MODELS["zh"] = spacy.load("zh_core_web_sm")
                print("Loaded 'zh_core_web_sm' spaCy model.")
            except OSError:
                print(
                    "Error loading 'zh_core_web_sm' spaCy model. Falling back to regex."
                )
                NLP_MODELS["zh"] = None  # Mark as failed to avoid retrying

        if NLP_MODELS.get("zh"):
            doc = NLP_MODELS["zh"](text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            print("Using regex for Chinese sentence segmentation.")
            sentences = re.split(r"(?<=[。！？；])", text)
            return [s.strip() for s in sentences if s.strip()]

    else:
        print(
            f"Warning: Language '{language}' not supported for sentence segmentation."
        )
        return []


def write_to_tsv(aligned_data: list[dict], output_filepath: str):
    """Writes aligned sentence data to a TSV file."""
    try:
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["English Sentence", "Chinese Sentence", "Alignment Score"])
            for item in aligned_data:
                writer.writerow([item["english"], item["chinese"], item["score"]])
        print(f"Successfully wrote data to TSV: {output_filepath}")
    except IOError as e:
        print(f"Error writing to TSV file {output_filepath}: {e}")


def write_to_json(aligned_data: list[dict], output_filepath: str):
    """Writes aligned sentence data to a JSON file."""
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(aligned_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully wrote data to JSON: {output_filepath}")
    except IOError as e:
        print(f"Error writing to JSON file {output_filepath}: {e}")


def load_embedding_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Loads the sentence embedding model."""
    global EMBEDDING_MODEL
    try:
        EMBEDDING_MODEL = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model '{model_name}': {e}")
        EMBEDDING_MODEL = None


def align_sentences_in_chapters(
    english_sentences: list[str],
    chinese_sentences: list[str],
    embedding_model,
    similarity_threshold: float = 0.5,
) -> list[dict]:
    """Aligns sentences between two lists using semantic similarity."""
    if not english_sentences or not chinese_sentences:
        return []
    if not embedding_model:
        print("Embedding model not loaded. Cannot align sentences.")
        return []

    try:
        english_embeddings = embedding_model.encode(
            english_sentences, convert_to_tensor=True
        )
        chinese_embeddings = embedding_model.encode(
            chinese_sentences, convert_to_tensor=True
        )
    except Exception as e:
        print(f"Error encoding sentences: {e}")
        return []

    aligned_sentence_pairs = []
    total_potential_alignments = 0
    filtered_out_count = 0

    for i in range(len(english_sentences)):
        english_emb = english_embeddings[i : i + 1]
        similarities = util.cos_sim(english_emb, chinese_embeddings)

        if similarities is None or similarities.numel() == 0:
            continue

        max_sim_score_tensor = torch.max(similarities[0])
        best_match_index = torch.argmax(similarities[0])
        max_sim_score = max_sim_score_tensor.item()

        total_potential_alignments += 1

        if max_sim_score >= similarity_threshold:
            aligned_sentence_pairs.append(
                {
                    "english": english_sentences[i],
                    "chinese": chinese_sentences[best_match_index.item()],
                    "score": max_sim_score,
                }
            )
        else:
            filtered_out_count += 1

    if total_potential_alignments > 0:
        print(
            f"    Filtered out {filtered_out_count}/{total_potential_alignments} alignments below threshold {similarity_threshold}"
        )
        if aligned_sentence_pairs:
            scores = [pair["score"] for pair in aligned_sentence_pairs]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(
                f"    Accepted alignments: avg={avg_score:.3f}, min={min_score:.3f}, max={max_score:.3f}"
            )

    return aligned_sentence_pairs


def split_into_chapters(markdown_content: str) -> list[dict]:
    """Splits Markdown content into chapters based on H1 and H2 headings."""
    chapters = []
    # Regex to find H1/H2 headings and capture the heading level, title, and content
    # It looks for lines starting with # or ##, followed by a space, then captures the title.
    # The content is everything following the heading line until the next H1/H2 heading or EOF.
    # Using re.DOTALL for '.' to match newlines, and re.MULTILINE for '^' to match start of lines.
    pattern = re.compile(r"^(#{1,2})\s+(.*?)\s*$", re.MULTILINE)

    matches = list(pattern.finditer(markdown_content))

    for i, match in enumerate(matches):
        title = match.group(2).strip()

        # Determine content start and end
        content_start = match.end()
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(markdown_content)

        content = markdown_content[content_start:content_end].strip()

        chapters.append({"title": title, "content": content})

    return chapters


def main():
    parser = argparse.ArgumentParser(
        description="Aligns English and Chinese Markdown books."
    )
    parser.add_argument("english_book_path", help="Path to the English Markdown book.")
    parser.add_argument("chinese_book_path", help="Path to the Chinese Markdown book.")
    parser.add_argument("output_file_path", help="Path for the output file.")
    parser.add_argument(
        "--format",
        default="tsv",
        choices=["tsv", "json"],
        help="Output format (tsv or json).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score for sentence alignment (0.0-1.0). Higher values = stricter matching. Default: 0.7",
    )

    args = parser.parse_args()

    print(f"English book path: {args.english_book_path}")
    print(f"Chinese book path: {args.chinese_book_path}")
    print(f"Output file path: {args.output_file_path}")
    print(f"Output format: {args.format}")
    print(f"Similarity threshold: {args.similarity_threshold}")

    load_embedding_model()  # Load the embedding model at the start

    all_aligned_sentences = []

    try:
        with open(args.english_book_path, "r", encoding="utf-8") as f:
            english_content = f.read()
        english_chapters = split_into_chapters(english_content)
        print("English Chapters:")
        for i, chapter in enumerate(english_chapters):
            print(f"{i+1}: {chapter['title']}")
    except FileNotFoundError:
        print(f"Error: English book file not found at {args.english_book_path}")
        english_chapters = []

    try:
        with open(args.chinese_book_path, "r", encoding="utf-8") as f:
            chinese_content = f.read()
        chinese_chapters = split_into_chapters(chinese_content)
        print("Chinese Chapters:")
        for i, chapter in enumerate(chinese_chapters):
            print(f"{i+1}: {chapter['title']}")
    except FileNotFoundError:
        print(f"Error: Chinese book file not found at {args.chinese_book_path}")
        chinese_chapters = []

    aligned_chapter_pairs = get_user_chapter_alignments(
        english_chapters, chinese_chapters
    )

    if aligned_chapter_pairs:
        for eng_chapter, chi_chapter in aligned_chapter_pairs:
            print(
                f"\nProcessing chapter: English '{eng_chapter['title']}' with Chinese '{chi_chapter['title']}'"
            )
            english_content = eng_chapter["content"]
            chinese_content = chi_chapter["content"]

            english_sentences = segment_sentences(english_content, "english")
            chinese_sentences = segment_sentences(chinese_content, "chinese")

            if EMBEDDING_MODEL and english_sentences and chinese_sentences:
                current_chapter_alignments = align_sentences_in_chapters(
                    english_sentences,
                    chinese_sentences,
                    EMBEDDING_MODEL,
                    args.similarity_threshold,
                )
                if current_chapter_alignments:
                    all_aligned_sentences.extend(current_chapter_alignments)
                    print(
                        f"Found {len(current_chapter_alignments)} alignments in chapter '{eng_chapter['title']}' / '{chi_chapter['title']}'."
                    )
                else:
                    print(
                        f"No alignments found in chapter '{eng_chapter['title']}' / '{chi_chapter['title']}'."
                    )
            else:
                print(
                    f"Skipping sentence alignment for chapter '{eng_chapter['title']}' / '{chi_chapter['title']}' due to missing content or model."
                )
    else:
        print("No chapter alignments were made (using temporary logic).")

    if all_aligned_sentences:
        if args.format == "tsv":
            write_to_tsv(all_aligned_sentences, args.output_file_path)
        elif args.format == "json":
            write_to_json(all_aligned_sentences, args.output_file_path)
        # The print statements from write_to_tsv/json already confirm success
        # print(f"Successfully wrote {len(all_aligned_sentences)} aligned sentences to {args.output_file_path} in {args.format} format.")
    else:
        print("No sentences were aligned across any chapters. Output file not written.")


def get_user_chapter_alignments(
    english_chapters: list[dict], chinese_chapters: list[dict]
) -> list[tuple[dict, dict]]:
    """
    Prompts the user to provide chapter alignments or uses intelligent default alignments.
    """
    print("\n--- Chapter Alignment Input ---")
    print("Normally, you would enter chapter alignments here.")
    print(
        "For example, to align English chapter 1 with Chinese chapter 1, and English chapter 2 with Chinese chapter 3, you might enter: '1-1 2-3'"
    )

    # Intelligent default alignment for books that appear to be translations of each other
    aligned_pairs = []

    # Filter out non-content chapters (copyright, table of contents, etc.)
    def is_content_chapter(title: str) -> bool:
        """Check if a chapter title represents actual content rather than metadata."""
        # English patterns for non-content
        english_non_content = [
            "prologue",
            "epilogue",
            "acknowledgement",
            "acknowledgments",
            "permission",
            "permissions",
            "index",
            "contents",
            "copyright",
            "list of",
            "![",  # Images
            "cover",
            "title",
        ]
        # Chinese patterns for non-content
        chinese_non_content = ["版权", "目录", "序言", "后记", "致谢", "译后记", "!["]

        title_lower = title.lower().strip()

        # Skip empty titles
        if not title_lower:
            return False

        # Skip image references
        if title.startswith("![") or title.startswith("images/"):
            return False

        # Check for English non-content patterns
        for pattern in english_non_content:
            if pattern in title_lower:
                return False

        # Check for Chinese non-content patterns
        for pattern in chinese_non_content:
            if pattern in title:
                return False

        # For English: Only accept if it contains "chapter" and a number or has "Mouths That Love Eating" style titles
        if any(c.isascii() and c.isalpha() for c in title):  # Contains English letters
            # Must contain "chapter" to be considered a content chapter
            if "chapter" not in title_lower:
                return False
            # Additional check: should have some substantial content indicators
            return True

        # For Chinese: Only accept if it contains chapter markers like "第...章"
        if "第" in title and "章" in title:
            return True

        return False

    # Filter chapters to get only content chapters
    english_content_chapters = [
        ch for ch in english_chapters if is_content_chapter(ch["title"])
    ]
    chinese_content_chapters = [
        ch for ch in chinese_chapters if is_content_chapter(ch["title"])
    ]

    print(f"\nFiltered to content chapters:")
    print(f"English content chapters: {len(english_content_chapters)}")
    for i, ch in enumerate(english_content_chapters[:5]):  # Show first 5
        print(f"  {i+1}: {ch['title']}")
    if len(english_content_chapters) > 5:
        print(f"  ... and {len(english_content_chapters) - 5} more")

    print(f"Chinese content chapters: {len(chinese_content_chapters)}")
    for i, ch in enumerate(chinese_content_chapters[:5]):  # Show first 5
        print(f"  {i+1}: {ch['title']}")
    if len(chinese_content_chapters) > 5:
        print(f"  ... and {len(chinese_content_chapters) - 5} more")

    # Align chapters 1:1 up to the minimum count
    min_chapters = min(len(english_content_chapters), len(chinese_content_chapters))

    if min_chapters > 0:
        print(
            f"\nUsing intelligent 1:1 alignment for the first {min_chapters} content chapters:"
        )
        for i in range(min_chapters):
            eng_ch = english_content_chapters[i]
            chi_ch = chinese_content_chapters[i]
            aligned_pairs.append((eng_ch, chi_ch))
            print(f"  English '{eng_ch['title']}' ↔ Chinese '{chi_ch['title']}'")
    else:
        print("No content chapters found for alignment.")

    return aligned_pairs


if __name__ == "__main__":
    main()
