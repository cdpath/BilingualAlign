"""
Bilingual Book Sentence Aligner

Aligns English and Chinese sentences from books in Markdown format using sentence embeddings.
"""

import re
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import csv
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ENGLISH_NON_CONTENT = [
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
    "![",
    "cover",
    "title",
    "preface",
    "foreword",
    "introduction",
    "about",
    "bibliography",
    "appendix",
    "glossary",
]

CHINESE_NON_CONTENT = [
    "版权",
    "目录",
    "序言",
    "后记",
    "致谢",
    "译后记",
    "![",
    "前言",
    "序",
    "跋",
    "附录",
    "索引",
]


@dataclass
class AlignmentConfig:
    """Configuration for the alignment process."""

    similarity_threshold: float = 0.7
    chapter_similarity_threshold: float = 0.35
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    english_model: str = "en_core_web_sm"
    chinese_model: str = "zh_core_web_sm"


@dataclass
class ChapterStats:
    """Statistics for chapter alignment."""

    total_english_chapters: int = 0
    total_chinese_chapters: int = 0
    matched_pairs: int = 0
    english_match_rate: float = 0.0
    chinese_match_rate: float = 0.0
    similarity_threshold: float = 0.35
    matched_chapters: List[Dict] = field(default_factory=list)
    unmatched_english: List[Dict] = field(default_factory=list)
    unmatched_chinese: List[Dict] = field(default_factory=list)


@dataclass
class ProcessingDetails:
    """Details for chapter processing."""

    english_chapter: str
    chinese_chapter: str
    english_sentences_count: int
    chinese_sentences_count: int
    aligned_sentences_count: int = 0
    sentence_match_rate: float = 0.0
    avg_similarity_score: float = 0.0


class ModelManager:
    """Manages NLP and embedding models - keeps state so class is justified."""

    def __init__(self, config: AlignmentConfig):
        self.config = config
        self._nlp_models: Dict[str, Optional[spacy.Language]] = {}
        self._embedding_model: Optional[SentenceTransformer] = None

    def _ensure_sentencizer(self, nlp: spacy.Language) -> spacy.Language:
        """Add spaCy sentencizer if dependency parser is missing."""
        if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    def get_nlp_model(self, language: str) -> Optional[spacy.Language]:
        """Get or load NLP model for the specified language."""
        model_key = language[:2]  # 'en' or 'zh'

        if model_key in self._nlp_models:
            return self._nlp_models[model_key]

        model_name = (
            self.config.english_model
            if language == "english"
            else self.config.chinese_model
        )

        try:
            nlp = spacy.load(model_name, exclude=["ner", "lemmatizer"])
            self._nlp_models[model_key] = self._ensure_sentencizer(nlp)
            logger.info(f"Loaded '{model_name}' spaCy model.")
            return self._nlp_models[model_key]
        except OSError:
            logger.error(f"Error loading '{model_name}' spaCy model.")
            logger.info(f"To install: python -m spacy download {model_name}")
            if language == "chinese":
                logger.info("Falling back to regex for Chinese sentence segmentation.")
                self._nlp_models[model_key] = None
                return None
            raise

    @property
    def embedding_model(self) -> Optional[SentenceTransformer]:
        """Get or load the embedding model."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model

    def _load_embedding_model(self) -> None:
        """Load the sentence embedding model."""
        try:
            # Prefer MPS (Apple Silicon) > CUDA > CPU
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self._embedding_model = SentenceTransformer(
                self.config.embedding_model_name, device=device
            )
            logger.info(
                f"Embedding model '{self.config.embedding_model_name}' loaded successfully on {device}."
            )
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{self.config.embedding_model_name}': {e}"
            )
            self._embedding_model = None


# File I/O functions - plain functions are better for stateless operations
def read_file(file_path: Union[str, Path]) -> str:
    """Read a file and return its content."""
    path = Path(file_path)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


def write_tsv(aligned_data: List[Dict], output_path: Union[str, Path]) -> None:
    """Write aligned sentence data to a TSV file."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["English Sentence", "Chinese Sentence", "Alignment Score"])
            for item in aligned_data:
                writer.writerow([item["english"], item["chinese"], item["score"]])
        logger.info(f"Successfully wrote data to TSV: {path}")
    except IOError as e:
        logger.error(f"Error writing to TSV file {path}: {e}")
        raise


def write_json(data: Union[List, Dict], output_path: Union[str, Path]) -> None:
    """Write data to a JSON file."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully wrote data to JSON: {path}")
    except IOError as e:
        logger.error(f"Error writing to JSON file {path}: {e}")
        raise


def write_alignment_report(report_data: Dict, output_path: Union[str, Path]) -> None:
    """Write a detailed matching report to a JSON file."""
    path = Path(output_path)
    report_path = path.parent / "alignment_report.json"
    write_json(report_data, report_path)


# Text processing functions
def segment_sentences(
    text: str, language: str, model_manager: ModelManager
) -> List[str]:
    """Segments text into sentences based on language."""
    if language == "english":
        nlp = model_manager.get_nlp_model("english")
        if nlp is None:
            return []

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    elif language == "chinese":
        nlp = model_manager.get_nlp_model("chinese")

        if nlp is not None:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            logger.info("Using regex for Chinese sentence segmentation.")
            # Keep punctuation, handle …… and Western full stop properly
            sentences = re.split(r"(?<=[。！？；…])\s*", text)
            return [s.strip() for s in sentences if s.strip()]

    else:
        logger.warning(
            f"Language '{language}' not supported for sentence segmentation."
        )
        return []


def split_into_chapters(markdown_content: str) -> List[Dict]:
    """Split Markdown content into chapters based on H1-H5 headings."""
    chapters = []
    # Accept # through ##### headings to handle Chinese markdown with ### chapters
    pattern = re.compile(r"^(#{1,5})\s+(.*?)\s*$", re.MULTILINE)

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


def is_content_chapter(title: str) -> bool:
    """Check if a chapter title represents actual content rather than metadata."""
    title_lower = title.lower().strip()

    # Skip empty titles
    if not title_lower:
        return False

    # Skip image references
    if title.startswith("![") or title.startswith("images/"):
        return False

    # Check for English non-content patterns
    for pattern in ENGLISH_NON_CONTENT:
        if pattern in title_lower:
            return False

    # Check for Chinese non-content patterns
    for pattern in CHINESE_NON_CONTENT:
        if pattern in title:
            return False

    # Enhanced English chapter detection with Roman numerals
    if any(c.isascii() and c.isalpha() for c in title):  # Contains English letters
        # Match "Chapter" followed by numbers or Roman numerals
        if re.search(r"\bchapter\s+(?:\d+|[IVXLCDM]+)\b", title_lower):
            return True
        # Match standalone Roman numerals (common chapter numbering)
        if re.search(r"^[IVXLCDM]+\.?\s*$", title.upper()):
            return True
        # Match "Part" followed by numbers or Roman numerals
        if re.search(r"\bpart\s+(?:\d+|[IVXLCDM]+)\b", title_lower):
            return True
        # Match simple numeric patterns like "1", "2.", "Chapter 1", etc.
        if re.search(r"^\d+\.?\s*$", title.strip()):
            return True
        return False

    # Enhanced Chinese chapter detection
    # Match 第...章/节/回/篇 patterns
    if re.search(r"第.+[章节回篇]", title):
        return True
    # Match standalone Chinese numerals followed by chapter indicators
    if re.search(r"[一二三四五六七八九十百千万]+[章节回篇]", title):
        return True
    # Match Arabic numerals with Chinese chapter indicators
    if re.search(r"\d+[章节回篇]", title):
        return True

    return False


def auto_align_chapters(
    eng_chapters: List[Dict],
    chi_chapters: List[Dict],
    model_manager: ModelManager,
    config: AlignmentConfig,
) -> Tuple[List[Tuple[int, int]], ChapterStats]:
    """
    Compute cosine similarity of whole-chapter embeddings and return
    a list of index pairs with a greedy/max-weight-matching algorithm.
    """
    if not eng_chapters or not chi_chapters:
        return [], ChapterStats()

    embedding_model = model_manager.embedding_model
    if not embedding_model:
        logger.error("Embedding model not available for chapter alignment")
        return [], ChapterStats()

    logger.info("Computing chapter-level embeddings for automatic alignment...")

    try:
        # Get content embeddings for all chapters
        eng_contents = [ch["content"] for ch in eng_chapters]
        chi_contents = [ch["content"] for ch in chi_chapters]

        with torch.no_grad():
            eng_vec = embedding_model.encode(eng_contents, convert_to_tensor=True)
            chi_vec = embedding_model.encode(chi_contents, convert_to_tensor=True)

        # Compute similarity matrix (E × C)
        sim = util.cos_sim(eng_vec, chi_vec)

        # Store original similarity matrix for reporting
        original_sim = sim.clone()

        pairs = []
        matched_details = []

        # Greedy matching: repeatedly find the highest similarity pair
        while True:
            # Find the maximum similarity in the remaining matrix
            idx = torch.argmax(sim)
            score = sim.flatten()[idx].item()

            if score < config.chapter_similarity_threshold:
                break

            # Convert flat index to row, column
            r, c = divmod(idx.item(), sim.size(1))
            pairs.append((r, c))

            matched_details.append(
                {
                    "english_index": r,
                    "english_title": eng_chapters[r]["title"],
                    "chinese_index": c,
                    "chinese_title": chi_chapters[c]["title"],
                    "similarity_score": score,
                }
            )

            logger.info(
                f"  Matched: EN '{eng_chapters[r]['title']}' ↔ CN '{chi_chapters[c]['title']}' (score: {score:.3f})"
            )

            # Mark this row and column as used by setting to -1
            sim[r, :] = -1.0
            sim[:, c] = -1.0

        # Find unmatched chapters
        matched_eng_indices = {pair[0] for pair in pairs}
        matched_chi_indices = {pair[1] for pair in pairs}

        unmatched_english = []
        for i, ch in enumerate(eng_chapters):
            if i not in matched_eng_indices:
                unmatched_english.append(
                    {
                        "index": i,
                        "title": ch["title"],
                        "best_match_score": original_sim[i].max().item(),
                        "best_match_chinese": chi_chapters[
                            original_sim[i].argmax().item()
                        ]["title"],
                    }
                )

        unmatched_chinese = []
        for i, ch in enumerate(chi_chapters):
            if i not in matched_chi_indices:
                unmatched_chinese.append(
                    {
                        "index": i,
                        "title": ch["title"],
                        "best_match_score": original_sim[:, i].max().item(),
                        "best_match_english": eng_chapters[
                            original_sim[:, i].argmax().item()
                        ]["title"],
                    }
                )

        # Calculate matching statistics
        alignment_stats = ChapterStats(
            total_english_chapters=len(eng_chapters),
            total_chinese_chapters=len(chi_chapters),
            matched_pairs=len(pairs),
            english_match_rate=len(pairs) / len(eng_chapters) if eng_chapters else 0,
            chinese_match_rate=len(pairs) / len(chi_chapters) if chi_chapters else 0,
            similarity_threshold=config.chapter_similarity_threshold,
            matched_chapters=matched_details,
            unmatched_english=unmatched_english,
            unmatched_chinese=unmatched_chinese,
        )

        # Sort pairs by English chapter index for consistent ordering
        pairs.sort()
        return pairs, alignment_stats

    except Exception as e:
        logger.error(f"Error in automatic chapter alignment: {e}")
        return [], ChapterStats()


# Sentence alignment functions
def align_sentences(
    english_sentences: List[str],
    chinese_sentences: List[str],
    model_manager: ModelManager,
    config: AlignmentConfig,
) -> List[Dict]:
    """Align sentences between two lists using semantic similarity."""
    if not english_sentences or not chinese_sentences:
        return []

    embedding_model = model_manager.embedding_model
    if not embedding_model:
        logger.error("Embedding model not loaded. Cannot align sentences.")
        return []

    try:
        with torch.no_grad():  # Save GPU memory
            english_embeddings = embedding_model.encode(
                english_sentences, convert_to_tensor=True
            )
            chinese_embeddings = embedding_model.encode(
                chinese_sentences, convert_to_tensor=True
            )
    except Exception as e:
        logger.error(f"Error encoding sentences: {e}")
        return []

    aligned_sentence_pairs = []
    total_potential_alignments = 0
    filtered_out_count = 0

    # Vectorize similarity computation - calculate all similarities at once
    similarities = util.cos_sim(english_embeddings, chinese_embeddings)
    used_chinese_indices = set()

    for i in range(len(english_sentences)):
        if similarities is None or similarities.numel() == 0:
            continue

        # Find best match for this English sentence
        row = similarities[i]
        best_match_index = torch.argmax(row)
        max_sim_score = row[best_match_index].item()
        best_idx = best_match_index.item()

        total_potential_alignments += 1

        # Only align if score is above threshold and Chinese sentence hasn't been used
        if (
            max_sim_score >= config.similarity_threshold
            and best_idx not in used_chinese_indices
        ):
            used_chinese_indices.add(best_idx)
            aligned_sentence_pairs.append(
                {
                    "english": english_sentences[i],
                    "chinese": chinese_sentences[best_idx],
                    "score": max_sim_score,
                }
            )
        else:
            filtered_out_count += 1

    if total_potential_alignments > 0:
        logger.info(
            f"    Filtered out {filtered_out_count}/{total_potential_alignments} "
            f"alignments below threshold {config.similarity_threshold}"
        )
        if aligned_sentence_pairs:
            scores = [pair["score"] for pair in aligned_sentence_pairs]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            logger.info(
                f"    Accepted alignments: avg={avg_score:.3f}, "
                f"min={min_score:.3f}, max={max_score:.3f}"
            )

    return aligned_sentence_pairs


class ChapterAlignmentManager:
    """Manages chapter alignment strategies - complex logic justifies class."""

    def __init__(self, model_manager: ModelManager, config: AlignmentConfig):
        self.model_manager = model_manager
        self.config = config

    def get_chapter_alignments(
        self,
        english_chapters: List[Dict],
        chinese_chapters: List[Dict],
        manual_align: Optional[str] = None,
        auto_align: bool = False,
    ) -> Tuple[List[Tuple[Dict, Dict]], Optional[ChapterStats]]:
        """Get chapter alignments using the specified strategy."""
        logger.info("\n--- Chapter Alignment Input ---")

        # Filter chapters to get only content chapters
        english_content_chapters = [
            ch for ch in english_chapters if is_content_chapter(ch["title"])
        ]
        chinese_content_chapters = [
            ch for ch in chinese_chapters if is_content_chapter(ch["title"])
        ]

        self._log_chapter_info(english_content_chapters, chinese_content_chapters)

        if manual_align and manual_align.strip():
            return (
                self._parse_manual_alignment(
                    manual_align, english_content_chapters, chinese_content_chapters
                ),
                None,
            )
        elif auto_align:
            return self._get_auto_alignment(
                english_content_chapters, chinese_content_chapters
            )
        else:
            return (
                self._get_default_alignment(
                    english_content_chapters, chinese_content_chapters
                ),
                None,
            )

    def _log_chapter_info(
        self, english_chapters: List[Dict], chinese_chapters: List[Dict]
    ) -> None:
        """Log information about filtered chapters."""
        logger.info("Filtered to content chapters:")
        logger.info(f"English content chapters: {len(english_chapters)}")
        for i, ch in enumerate(english_chapters[:5]):  # Show first 5
            logger.info(f"  {i + 1}: {ch['title']}")
        if len(english_chapters) > 5:
            logger.info(f"  ... and {len(english_chapters) - 5} more")

        logger.info(f"Chinese content chapters: {len(chinese_chapters)}")
        for i, ch in enumerate(chinese_chapters[:5]):  # Show first 5
            logger.info(f"  {i + 1}: {ch['title']}")
        if len(chinese_chapters) > 5:
            logger.info(f"  ... and {len(chinese_chapters) - 5} more")

    def _parse_manual_alignment(
        self,
        manual_align: str,
        english_chapters: List[Dict],
        chinese_chapters: List[Dict],
    ) -> List[Tuple[Dict, Dict]]:
        """Parse manual alignment string."""
        logger.info(f"Using manual alignment: {manual_align}")
        aligned_pairs = []

        try:
            for pair in manual_align.split():
                if "-" not in pair:
                    logger.warning(
                        f"Invalid pair format '{pair}'. Expected format: 'eng_idx-chi_idx'. Skipping."
                    )
                    continue

                eng_part, chi_part = pair.split("-", 1)

                # Handle skip notation (0 means skip)
                if eng_part == "0" or chi_part == "0":
                    logger.info(f"Skipping pair {pair} (contains 0)")
                    continue

                # Parse and validate indices
                try:
                    eng_indices = [int(x.strip()) for x in eng_part.split(",")]
                    chi_indices = [int(x.strip()) for x in chi_part.split(",")]
                except ValueError:
                    logger.warning(f"Invalid indices in pair {pair}. Skipping.")
                    continue

                # Collect and merge chapters
                eng_chapters_for_pair = self._get_chapters_by_indices(
                    eng_indices, english_chapters, "English", pair
                )
                chi_chapters_for_pair = self._get_chapters_by_indices(
                    chi_indices, chinese_chapters, "Chinese", pair
                )

                if eng_chapters_for_pair and chi_chapters_for_pair:
                    eng_merged = self._merge_chapters(eng_chapters_for_pair)
                    chi_merged = self._merge_chapters(chi_chapters_for_pair)

                    aligned_pairs.append((eng_merged, chi_merged))
                    logger.info(
                        f"  Aligned: '{eng_merged['title']}' ↔ '{chi_merged['title']}'"
                    )

        except Exception as e:
            logger.error(
                f"Error parsing manual alignment '{manual_align}': {e}. Falling back to defaults."
            )
            return self._get_default_alignment(english_chapters, chinese_chapters)

        return aligned_pairs

    def _get_chapters_by_indices(
        self, indices: List[int], chapters: List[Dict], language: str, pair_name: str
    ) -> List[Dict]:
        """Get chapters by their indices."""
        result_chapters = []
        for idx in indices:
            if 1 <= idx <= len(chapters):
                result_chapters.append(chapters[idx - 1])
            else:
                logger.warning(
                    f"{language} index {idx} out of range in pair {pair_name}. Skipping pair."
                )
                return []
        return result_chapters

    def _merge_chapters(self, chapters: List[Dict]) -> Dict:
        """Merge multiple chapters into one."""
        return {
            "title": " + ".join(ch["title"] for ch in chapters),
            "content": "\n\n".join(ch["content"] for ch in chapters),
        }

    def _get_auto_alignment(
        self, english_chapters: List[Dict], chinese_chapters: List[Dict]
    ) -> Tuple[List[Tuple[Dict, Dict]], ChapterStats]:
        """Get automatic chapter alignment."""
        logger.info("Using automatic chapter alignment based on content similarity...")
        idx_pairs, chapter_stats = auto_align_chapters(
            english_chapters, chinese_chapters, self.model_manager, self.config
        )
        aligned_pairs = [
            (english_chapters[i], chinese_chapters[j]) for i, j in idx_pairs
        ]

        if aligned_pairs:
            logger.info(
                f"Automatic alignment found {len(aligned_pairs)} chapter pairs:"
            )
            for eng_ch, chi_ch in aligned_pairs:
                logger.info(f"  '{eng_ch['title']}' ↔ '{chi_ch['title']}'")
        else:
            logger.info("No suitable automatic alignments found.")

        return aligned_pairs, chapter_stats

    def _get_default_alignment(
        self, english_chapters: List[Dict], chinese_chapters: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """Get default 1:1 alignment."""
        aligned_pairs = []
        min_chapters = min(len(english_chapters), len(chinese_chapters))

        if min_chapters > 0:
            logger.info(
                f"Using intelligent 1:1 alignment for the first {min_chapters} content chapters:"
            )
            for i in range(min_chapters):
                eng_ch = english_chapters[i]
                chi_ch = chinese_chapters[i]
                aligned_pairs.append((eng_ch, chi_ch))
                logger.info(
                    f"  English '{eng_ch['title']}' ↔ Chinese '{chi_ch['title']}'"
                )
        else:
            logger.info("No content chapters found for alignment.")

        return aligned_pairs


class BookAligner:
    """Main class for aligning bilingual books - orchestration justifies class."""

    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        self.model_manager = ModelManager(self.config)
        self.alignment_manager = ChapterAlignmentManager(
            self.model_manager, self.config
        )

        # State for reporting
        self.chapter_alignment_stats: Optional[ChapterStats] = None

    def process_books(
        self,
        english_book_path: Union[str, Path],
        chinese_book_path: Union[str, Path],
        output_file_path: Union[str, Path],
        output_format: str = "tsv",
        manual_align: Optional[str] = None,
        auto_align: bool = False,
    ) -> bool:
        """Main processing function that aligns sentences from two books."""
        english_path = Path(english_book_path)
        chinese_path = Path(chinese_book_path)
        output_path = Path(output_file_path)

        logger.info(f"English book path: {english_path}")
        logger.info(f"Chinese book path: {chinese_path}")
        logger.info(f"Output file path: {output_path}")
        logger.info(f"Output format: {output_format}")
        logger.info(f"Similarity threshold: {self.config.similarity_threshold}")

        try:
            # Read and process books
            english_content = read_file(english_path)
            chinese_content = read_file(chinese_path)

            english_chapters = split_into_chapters(english_content)
            chinese_chapters = split_into_chapters(chinese_content)

            self._log_chapters(english_chapters, "English")
            self._log_chapters(chinese_chapters, "Chinese")

            # Get chapter alignments
            aligned_chapter_pairs, chapter_stats = (
                self.alignment_manager.get_chapter_alignments(
                    english_chapters, chinese_chapters, manual_align, auto_align
                )
            )

            # Store chapter alignment stats for reporting
            self.chapter_alignment_stats = chapter_stats

            if not aligned_chapter_pairs:
                logger.error("No chapter alignments were made.")
                return False

            # Process aligned chapters
            all_aligned_sentences, processing_details = self._process_aligned_chapters(
                aligned_chapter_pairs
            )

            if all_aligned_sentences:
                # Write output files
                self._write_output_files(
                    all_aligned_sentences,
                    output_path,
                    output_format,
                    processing_details,
                    english_path,
                    chinese_path,
                )
                return True
            else:
                logger.error(
                    "No sentences were aligned across any chapters. Output file not written."
                )
                return False

        except Exception as e:
            logger.error(f"Error processing books: {e}")
            return False

    def _log_chapters(self, chapters: List[Dict], language: str) -> None:
        """Log chapter information."""
        logger.info(f"{language} Chapters:")
        for i, chapter in enumerate(chapters):
            logger.info(f"{i + 1}: {chapter['title']}")

    def _process_aligned_chapters(
        self, aligned_chapter_pairs: List[Tuple[Dict, Dict]]
    ) -> Tuple[List[Dict], List[ProcessingDetails]]:
        """Process aligned chapter pairs to extract sentence alignments."""
        all_aligned_sentences = []
        processing_details = []

        for eng_chapter, chi_chapter in aligned_chapter_pairs:
            logger.info(
                f"\nProcessing chapter: English '{eng_chapter['title']}' "
                f"with Chinese '{chi_chapter['title']}'"
            )

            # Segment sentences
            english_sentences = segment_sentences(
                eng_chapter["content"], "english", self.model_manager
            )
            chinese_sentences = segment_sentences(
                chi_chapter["content"], "chinese", self.model_manager
            )

            # Create processing detail record
            detail = ProcessingDetails(
                english_chapter=eng_chapter["title"],
                chinese_chapter=chi_chapter["title"],
                english_sentences_count=len(english_sentences),
                chinese_sentences_count=len(chinese_sentences),
            )

            # Align sentences if we have content
            if english_sentences and chinese_sentences:
                current_chapter_alignments = align_sentences(
                    english_sentences,
                    chinese_sentences,
                    self.model_manager,
                    self.config,
                )

                if current_chapter_alignments:
                    all_aligned_sentences.extend(current_chapter_alignments)
                    detail.aligned_sentences_count = len(current_chapter_alignments)
                    detail.sentence_match_rate = len(current_chapter_alignments) / min(
                        len(english_sentences), len(chinese_sentences)
                    )
                    detail.avg_similarity_score = sum(
                        pair["score"] for pair in current_chapter_alignments
                    ) / len(current_chapter_alignments)
                    logger.info(
                        f"Found {len(current_chapter_alignments)} alignments in chapter "
                        f"'{eng_chapter['title']}' / '{chi_chapter['title']}'."
                    )
                else:
                    logger.info(
                        f"No alignments found in chapter '{eng_chapter['title']}' / "
                        f"'{chi_chapter['title']}'."
                    )
            else:
                logger.info(
                    f"Skipping sentence alignment for chapter '{eng_chapter['title']}' / "
                    f"'{chi_chapter['title']}' due to missing content."
                )

            processing_details.append(detail)

        return all_aligned_sentences, processing_details

    def _write_output_files(
        self,
        aligned_sentences: List[Dict],
        output_path: Path,
        output_format: str,
        processing_details: List[ProcessingDetails],
        english_path: Path,
        chinese_path: Path,
    ) -> None:
        """Write output files and generate reports."""
        # Write main output file
        if output_format == "tsv":
            write_tsv(aligned_sentences, output_path)
        elif output_format == "json":
            write_json(aligned_sentences, output_path)

        # Generate and write comprehensive report
        report_data = self._generate_report(
            aligned_sentences,
            processing_details,
            english_path,
            chinese_path,
            output_path,
            output_format,
        )
        write_alignment_report(report_data, output_path)

        # Print summary
        self._print_summary(report_data)

    def _generate_report(
        self,
        aligned_sentences: List[Dict],
        processing_details: List[ProcessingDetails],
        english_path: Path,
        chinese_path: Path,
        output_path: Path,
        output_format: str,
    ) -> Dict:
        """Generate comprehensive alignment report."""
        total_eng_sentences = sum(
            detail.english_sentences_count for detail in processing_details
        )
        total_chi_sentences = sum(
            detail.chinese_sentences_count for detail in processing_details
        )
        total_aligned_sentences = len(aligned_sentences)

        return {
            "processing_summary": {
                "total_english_sentences": total_eng_sentences,
                "total_chinese_sentences": total_chi_sentences,
                "total_aligned_sentences": total_aligned_sentences,
                "overall_sentence_match_rate": (
                    total_aligned_sentences
                    / min(total_eng_sentences, total_chi_sentences)
                    if total_eng_sentences and total_chi_sentences
                    else 0
                ),
                "avg_alignment_score": (
                    sum(pair["score"] for pair in aligned_sentences)
                    / len(aligned_sentences)
                    if aligned_sentences
                    else 0
                ),
                "similarity_threshold": self.config.similarity_threshold,
            },
            "chapter_alignment_stats": (
                self.chapter_alignment_stats.__dict__
                if self.chapter_alignment_stats
                else {}
            ),
            "chapter_processing_details": [
                detail.__dict__ for detail in processing_details
            ],
            "metadata": {
                "english_book_path": str(english_path),
                "chinese_book_path": str(chinese_path),
                "output_file_path": str(output_path),
                "output_format": output_format,
            },
        }

    def _print_summary(self, report_data: Dict) -> None:
        """Print alignment summary statistics."""
        processing_summary = report_data["processing_summary"]
        chapter_alignment_stats = report_data.get("chapter_alignment_stats", {})

        logger.info("=== ALIGNMENT SUMMARY ===")
        logger.info("Chapter-level matching:")

        if chapter_alignment_stats:
            logger.info(
                f"  English chapters: {chapter_alignment_stats.get('total_english_chapters', 0)}"
            )
            logger.info(
                f"  Chinese chapters: {chapter_alignment_stats.get('total_chinese_chapters', 0)}"
            )
            logger.info(
                f"  Matched chapter pairs: {chapter_alignment_stats.get('matched_pairs', 0)}"
            )
            logger.info(
                f"  English chapter match rate: {chapter_alignment_stats.get('english_match_rate', 0):.1%}"
            )
            logger.info(
                f"  Chinese chapter match rate: {chapter_alignment_stats.get('chinese_match_rate', 0):.1%}"
            )

            if chapter_alignment_stats.get("unmatched_english"):
                logger.info("  Unmatched English chapters:")
                for ch in chapter_alignment_stats["unmatched_english"]:
                    logger.info(
                        f"    - {ch['title']} (best match: {ch['best_match_chinese']}, score: {ch['best_match_score']:.3f})"
                    )

            if chapter_alignment_stats.get("unmatched_chinese"):
                logger.info("  Unmatched Chinese chapters:")
                for ch in chapter_alignment_stats["unmatched_chinese"]:
                    logger.info(
                        f"    - {ch['title']} (best match: {ch['best_match_english']}, score: {ch['best_match_score']:.3f})"
                    )

        logger.info("Sentence-level matching:")
        logger.info(
            f"  Total English sentences: {processing_summary['total_english_sentences']}"
        )
        logger.info(
            f"  Total Chinese sentences: {processing_summary['total_chinese_sentences']}"
        )
        logger.info(
            f"  Total aligned sentence pairs: {processing_summary['total_aligned_sentences']}"
        )
        logger.info(
            f"  Overall sentence match rate: {processing_summary['overall_sentence_match_rate']:.1%}"
        )
        logger.info(
            f"  Average alignment score: {processing_summary['avg_alignment_score']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Aligns English and Chinese sentences from books in Markdown format using sentence embeddings."
    )
    parser.add_argument(
        "english_book_path",
        help="Path to the English book in Markdown format",
    )
    parser.add_argument(
        "chinese_book_path",
        help="Path to the Chinese book in Markdown format",
    )
    parser.add_argument(
        "output_file_path",
        help="Path for the output file (TSV or JSON format)",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format: 'tsv' (tab-separated values) or 'json'. Default: tsv",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score for sentence alignment (0.0-1.0). Higher values = stricter matching. Default: 0.7",
    )
    parser.add_argument(
        "--align",
        help='Manual chapter mapping, e.g. "1-1 2-3 4-0". '
        '"0" on either side means "skip this chapter". '
        'Supports m-to-n mapping like "3-4,5" (EN 3 ↔ CN 4+5).',
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="Use automatic chapter alignment based on content similarity instead of positional matching.",
    )

    args = parser.parse_args()

    config = AlignmentConfig(similarity_threshold=args.similarity_threshold)
    aligner = BookAligner(config)

    success = aligner.process_books(
        english_book_path=args.english_book_path,
        chinese_book_path=args.chinese_book_path,
        output_file_path=args.output_file_path,
        output_format=args.format,
        manual_align=args.align,
        auto_align=args.auto_align,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
