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

__version__ = "0.1.0"

NLP_MODELS = {}
EMBEDDING_MODEL = None
chapter_alignment_stats = {}


def _ensure_sentencizer(nlp):
    """Add spaCy sentencizer if dependency parser is missing."""
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def segment_sentences(text: str, language: str) -> list[str]:
    """Segments text into sentences based on language."""
    if language == "english":
        if "en" not in NLP_MODELS:
            try:
                nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
                NLP_MODELS["en"] = _ensure_sentencizer(nlp)
                print("Loaded 'en_core_web_sm' spaCy model.")
            except OSError:
                print(
                    "Error loading 'en_core_web_sm' spaCy model."
                )
                print("To install: python -m spacy download en_core_web_sm")
                raise  # Raise the exception instead of falling back

        doc = NLP_MODELS["en"](text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    elif language == "chinese":
        if "zh" not in NLP_MODELS:
            try:
                nlp = spacy.load("zh_core_web_sm", exclude=["ner", "lemmatizer"])
                NLP_MODELS["zh"] = _ensure_sentencizer(nlp)
                print("Loaded 'zh_core_web_sm' spaCy model.")
            except OSError:
                print(
                    "Error loading 'zh_core_web_sm' spaCy model. Falling back to regex."
                )
                print("To install: python -m spacy download zh_core_web_sm")
                NLP_MODELS["zh"] = None  # Mark as failed to avoid retrying

        if NLP_MODELS.get("zh"):
            doc = NLP_MODELS["zh"](text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            print("Using regex for Chinese sentence segmentation.")
            # Keep punctuation, handle …… and Western full stop properly
            sentences = re.split(r"(?<=[。！？；…])\s*", text)
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


def write_match_report(report_data: dict, output_filepath: str):
    """Writes a detailed matching report to a JSON file."""
    try:
        output_dir = output_filepath.rsplit('/', 1)[0] if '/' in output_filepath else '.'
        report_path = f"{output_dir}/alignment_report.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully wrote alignment report: {report_path}")
    except IOError as e:
        print(f"Error writing alignment report: {e}")


def load_embedding_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Loads the sentence embedding model."""
    global EMBEDDING_MODEL
    try:
        # Prefer MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        EMBEDDING_MODEL = SentenceTransformer(model_name, device=device)
        print(f"Embedding model '{model_name}' loaded successfully on {device}.")
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
        with torch.no_grad():  # Save GPU memory
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
        if max_sim_score >= similarity_threshold and best_idx not in used_chinese_indices:
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
    """Splits Markdown content into chapters based on H1-H5 headings."""
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


def auto_align_chapters(eng_chapters: list[dict], chi_chapters: list[dict]) -> tuple[list[tuple[int, int]], dict]:
    """
    Compute cosine similarity of whole-chapter embeddings and return
    a list of index pairs with a greedy/max-weight-matching algorithm.
    
    Args:
        eng_chapters: List of English chapter dictionaries
        chi_chapters: List of Chinese chapter dictionaries
        
    Returns:
        Tuple of:
        - List of (eng_idx, chi_idx) tuples representing optimal alignments
        - Dictionary with detailed alignment statistics
    """
    if not eng_chapters or not chi_chapters or not EMBEDDING_MODEL:
        return [], {}
    
    print("Computing chapter-level embeddings for automatic alignment...")
    
    try:
        # Get content embeddings for all chapters
        eng_contents = [ch['content'] for ch in eng_chapters]
        chi_contents = [ch['content'] for ch in chi_chapters]
        
        with torch.no_grad():
            eng_vec = EMBEDDING_MODEL.encode(eng_contents, convert_to_tensor=True)
            chi_vec = EMBEDDING_MODEL.encode(chi_contents, convert_to_tensor=True)
        
        # Compute similarity matrix (E × C)
        sim = util.cos_sim(eng_vec, chi_vec)
        
        # Store original similarity matrix for reporting
        original_sim = sim.clone()
        
        pairs = []
        similarity_threshold = 0.35  # Adjustable cut-off for chapter similarity
        matched_details = []
        
        # Greedy matching: repeatedly find the highest similarity pair
        while True:
            # Find the maximum similarity in the remaining matrix
            idx = torch.argmax(sim)
            score = sim.flatten()[idx].item()
            
            if score < similarity_threshold:
                break
                
            # Convert flat index to row, column
            r, c = divmod(idx.item(), sim.size(1))
            pairs.append((r, c))
            
            matched_details.append({
                "english_index": r,
                "english_title": eng_chapters[r]['title'],
                "chinese_index": c, 
                "chinese_title": chi_chapters[c]['title'],
                "similarity_score": score
            })
            
            print(f"  Matched: EN '{eng_chapters[r]['title']}' ↔ CN '{chi_chapters[c]['title']}' (score: {score:.3f})")
            
            # Mark this row and column as used by setting to -1
            sim[r, :] = -1.0
            sim[:, c] = -1.0
        
        # Find unmatched chapters
        matched_eng_indices = {pair[0] for pair in pairs}
        matched_chi_indices = {pair[1] for pair in pairs}
        
        unmatched_english = []
        for i, ch in enumerate(eng_chapters):
            if i not in matched_eng_indices:
                unmatched_english.append({
                    "index": i,
                    "title": ch['title'],
                    "best_match_score": original_sim[i].max().item(),
                    "best_match_chinese": chi_chapters[original_sim[i].argmax().item()]['title']
                })
        
        unmatched_chinese = []
        for i, ch in enumerate(chi_chapters):
            if i not in matched_chi_indices:
                unmatched_chinese.append({
                    "index": i,
                    "title": ch['title'],
                    "best_match_score": original_sim[:, i].max().item(),
                    "best_match_english": eng_chapters[original_sim[:, i].argmax().item()]['title']
                })
        
        # Calculate matching statistics
        alignment_stats = {
            "total_english_chapters": len(eng_chapters),
            "total_chinese_chapters": len(chi_chapters),
            "matched_pairs": len(pairs),
            "english_match_rate": len(pairs) / len(eng_chapters) if eng_chapters else 0,
            "chinese_match_rate": len(pairs) / len(chi_chapters) if chi_chapters else 0,
            "similarity_threshold": similarity_threshold,
            "matched_chapters": matched_details,
            "unmatched_english": unmatched_english,
            "unmatched_chinese": unmatched_chinese
        }
        
        # Sort pairs by English chapter index for consistent ordering
        pairs.sort()
        return pairs, alignment_stats
        
    except Exception as e:
        print(f"Error in automatic chapter alignment: {e}")
        return [], {}


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
        "preface",
        "foreword",
        "introduction",
        "about",
        "bibliography",
        "appendix",
        "glossary",
    ]
    # Chinese patterns for non-content
    chinese_non_content = ["版权", "目录", "序言", "后记", "致谢", "译后记", "![", "前言", "序", "跋", "附录", "索引"]

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


def get_user_chapter_alignments(
    english_chapters: list[dict], chinese_chapters: list[dict], manual_align: str = None, auto_align: bool = False
) -> list[tuple[dict, dict]]:
    """
    Prompts the user to provide chapter alignments or uses intelligent default alignments.
    
    Args:
        english_chapters: List of English chapter dictionaries
        chinese_chapters: List of Chinese chapter dictionaries  
        manual_align: Manual alignment string like "1-1 2-3 4-0"
        auto_align: Whether to use automatic alignment based on content similarity
    """
    print("\n--- Chapter Alignment Input ---")
    
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

    aligned_pairs = []

    # Handle manual alignment string
    if manual_align and manual_align.strip():
        print(f"\nUsing manual alignment: {manual_align}")
        try:
            for pair in manual_align.split():
                if '-' not in pair:
                    print(f"Invalid pair format '{pair}'. Expected format: 'eng_idx-chi_idx'. Skipping.")
                    continue
                    
                eng_part, chi_part = pair.split('-', 1)
                
                # Handle skip notation (0 means skip)
                if eng_part == '0' or chi_part == '0':
                    print(f"Skipping pair {pair} (contains 0)")
                    continue
                
                # Parse English indices (support single or comma-separated)
                try:
                    eng_indices = [int(x.strip()) for x in eng_part.split(',')]
                except ValueError:
                    print(f"Invalid English indices in pair {pair}. Skipping.")
                    continue
                
                # Parse Chinese indices (support single or comma-separated)  
                try:
                    chi_indices = [int(x.strip()) for x in chi_part.split(',')]
                except ValueError:
                    print(f"Invalid Chinese indices in pair {pair}. Skipping.")
                    continue
                
                # Validate indices and collect chapters
                eng_chapters_for_pair = []
                for idx in eng_indices:
                    if 1 <= idx <= len(english_content_chapters):
                        eng_chapters_for_pair.append(english_content_chapters[idx-1])
                    else:
                        print(f"English index {idx} out of range in pair {pair}. Skipping pair.")
                        eng_chapters_for_pair = []
                        break
                
                chi_chapters_for_pair = []
                for idx in chi_indices:
                    if 1 <= idx <= len(chinese_content_chapters):
                        chi_chapters_for_pair.append(chinese_content_chapters[idx-1])
                    else:
                        print(f"Chinese index {idx} out of range in pair {pair}. Skipping pair.")
                        chi_chapters_for_pair = []
                        break
                
                # If we have valid chapters, create merged chapter objects
                if eng_chapters_for_pair and chi_chapters_for_pair:
                    # Merge English chapters
                    eng_merged = {
                        "title": " + ".join(ch["title"] for ch in eng_chapters_for_pair),
                        "content": "\n\n".join(ch["content"] for ch in eng_chapters_for_pair)
                    }
                    
                    # Merge Chinese chapters  
                    chi_merged = {
                        "title": " + ".join(ch["title"] for ch in chi_chapters_for_pair),
                        "content": "\n\n".join(ch["content"] for ch in chi_chapters_for_pair)
                    }
                    
                    aligned_pairs.append((eng_merged, chi_merged))
                    print(f"  Aligned: '{eng_merged['title']}' ↔ '{chi_merged['title']}'")
                    
        except Exception as e:
            print(f"Error parsing manual alignment '{manual_align}': {e}. Falling back to defaults.")
            aligned_pairs = []

    # Handle automatic alignment
    elif auto_align:
        print("\nUsing automatic chapter alignment based on content similarity...")
        idx_pairs, chapter_stats = auto_align_chapters(english_content_chapters, chinese_content_chapters)
        aligned_pairs = [(english_content_chapters[i], chinese_content_chapters[j]) for i, j in idx_pairs]
        
        if aligned_pairs:
            print(f"\nAutomatic alignment found {len(aligned_pairs)} chapter pairs:")
            for eng_ch, chi_ch in aligned_pairs:
                print(f"  '{eng_ch['title']}' ↔ '{chi_ch['title']}'")
        else:
            print("No suitable automatic alignments found.")
        
        # Store chapter alignment statistics for later use
        global chapter_alignment_stats
        chapter_alignment_stats = chapter_stats

    # Interactive mode (original behavior)
    else:
        print("Normally, you would enter chapter alignments here.")
        print(
            "For example, to align English chapter 1 with Chinese chapter 1, and English chapter 2 with Chinese chapter 3, you might enter: '1-1 2-3'"
        )
        print("For m-to-n mapping, use commas: '3-4,5' means EN 3 ↔ CN 4+5")
        print("Use '0' to skip: '3-0' means skip English chapter 3")

        # Get user input for chapter alignments
        raw = input("\nEnter pairs (e.g. 1-1 2-3) or press ENTER to accept defaults: ")
        if raw.strip():
            # Reuse the manual alignment parsing logic
            return get_user_chapter_alignments(english_chapters, chinese_chapters, manual_align=raw.strip())

    # If no alignments were made, use intelligent 1:1 alignment as fallback
    if not aligned_pairs:
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


def process_books(
    english_book_path: str,
    chinese_book_path: str,
    output_file_path: str,
    output_format: str = "tsv",
    similarity_threshold: float = 0.7,
    manual_align: str = None,
    auto_align: bool = False,
) -> bool:
    """
    Main processing function that aligns sentences from two books.
    """
    print(f"English book path: {english_book_path}")
    print(f"Chinese book path: {chinese_book_path}")
    print(f"Output file path: {output_file_path}")
    print(f"Output format: {output_format}")
    print(f"Similarity threshold: {similarity_threshold}")

    load_embedding_model()

    all_aligned_sentences = []
    chapter_processing_details = []

    try:
        with open(english_book_path, "r", encoding="utf-8") as f:
            english_book_content = f.read()
        english_chapters = split_into_chapters(english_book_content)
        print("English Chapters:")
        for i, chapter in enumerate(english_chapters):
            print(f"{i+1}: {chapter['title']}")
    except FileNotFoundError:
        print(f"Error: English book file not found at {english_book_path}")
        return False

    try:
        with open(chinese_book_path, "r", encoding="utf-8") as f:
            chinese_book_content = f.read()
        chinese_chapters = split_into_chapters(chinese_book_content)
        print("Chinese Chapters:")
        for i, chapter in enumerate(chinese_chapters):
            print(f"{i+1}: {chapter['title']}")
    except FileNotFoundError:
        print(f"Error: Chinese book file not found at {chinese_book_path}")
        return False

    aligned_chapter_pairs = get_user_chapter_alignments(
        english_chapters, chinese_chapters, manual_align=manual_align, auto_align=auto_align
    )

    if aligned_chapter_pairs:
        for eng_chapter, chi_chapter in aligned_chapter_pairs:
            print(
                f"\nProcessing chapter: English '{eng_chapter['title']}' with Chinese '{chi_chapter['title']}'"
            )
            # Use different variable names to avoid shadowing
            eng_chapter_content = eng_chapter["content"]
            chi_chapter_content = chi_chapter["content"]

            english_sentences = segment_sentences(eng_chapter_content, "english")
            chinese_sentences = segment_sentences(chi_chapter_content, "chinese")

            chapter_detail = {
                "english_chapter": eng_chapter['title'],
                "chinese_chapter": chi_chapter['title'],
                "english_sentences_count": len(english_sentences),
                "chinese_sentences_count": len(chinese_sentences),
                "aligned_sentences_count": 0,
                "sentence_match_rate": 0.0,
                "avg_similarity_score": 0.0
            }

            if EMBEDDING_MODEL and english_sentences and chinese_sentences:
                current_chapter_alignments = align_sentences_in_chapters(
                    english_sentences,
                    chinese_sentences,
                    EMBEDDING_MODEL,
                    similarity_threshold,
                )
                if current_chapter_alignments:
                    all_aligned_sentences.extend(current_chapter_alignments)
                    chapter_detail["aligned_sentences_count"] = len(current_chapter_alignments)
                    chapter_detail["sentence_match_rate"] = len(current_chapter_alignments) / min(len(english_sentences), len(chinese_sentences))
                    chapter_detail["avg_similarity_score"] = sum(pair["score"] for pair in current_chapter_alignments) / len(current_chapter_alignments)
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
            
            chapter_processing_details.append(chapter_detail)
    else:
        print("No chapter alignments were made.")
        return False

    # Generate comprehensive alignment report
    global chapter_alignment_stats
    total_eng_sentences = sum(detail["english_sentences_count"] for detail in chapter_processing_details)
    total_chi_sentences = sum(detail["chinese_sentences_count"] for detail in chapter_processing_details)
    total_aligned_sentences = len(all_aligned_sentences)
    
    overall_stats = {
        "processing_summary": {
            "total_english_sentences": total_eng_sentences,
            "total_chinese_sentences": total_chi_sentences,
            "total_aligned_sentences": total_aligned_sentences,
            "overall_sentence_match_rate": total_aligned_sentences / min(total_eng_sentences, total_chi_sentences) if total_eng_sentences and total_chi_sentences else 0,
            "avg_alignment_score": sum(pair["score"] for pair in all_aligned_sentences) / len(all_aligned_sentences) if all_aligned_sentences else 0,
            "similarity_threshold": similarity_threshold
        },
        "chapter_alignment_stats": chapter_alignment_stats,
        "chapter_processing_details": chapter_processing_details,
        "metadata": {
            "english_book_path": english_book_path,
            "chinese_book_path": chinese_book_path,
            "output_file_path": output_file_path,
            "output_format": output_format
        }
    }

    # Print summary statistics
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"Chapter-level matching:")
    if chapter_alignment_stats:
        print(f"  English chapters: {chapter_alignment_stats.get('total_english_chapters', 0)}")
        print(f"  Chinese chapters: {chapter_alignment_stats.get('total_chinese_chapters', 0)}")
        print(f"  Matched chapter pairs: {chapter_alignment_stats.get('matched_pairs', 0)}")
        print(f"  English chapter match rate: {chapter_alignment_stats.get('english_match_rate', 0):.1%}")
        print(f"  Chinese chapter match rate: {chapter_alignment_stats.get('chinese_match_rate', 0):.1%}")
        
        if chapter_alignment_stats.get('unmatched_english'):
            print(f"\n  Unmatched English chapters:")
            for ch in chapter_alignment_stats['unmatched_english']:
                print(f"    - {ch['title']} (best match: {ch['best_match_chinese']}, score: {ch['best_match_score']:.3f})")
        
        if chapter_alignment_stats.get('unmatched_chinese'):
            print(f"\n  Unmatched Chinese chapters:")
            for ch in chapter_alignment_stats['unmatched_chinese']:
                print(f"    - {ch['title']} (best match: {ch['best_match_english']}, score: {ch['best_match_score']:.3f})")
    
    print(f"\nSentence-level matching:")
    print(f"  Total English sentences: {total_eng_sentences}")
    print(f"  Total Chinese sentences: {total_chi_sentences}")
    print(f"  Total aligned sentence pairs: {total_aligned_sentences}")
    print(f"  Overall sentence match rate: {overall_stats['processing_summary']['overall_sentence_match_rate']:.1%}")
    print(f"  Average alignment score: {overall_stats['processing_summary']['avg_alignment_score']:.3f}")

    if all_aligned_sentences:
        if output_format == "tsv":
            write_to_tsv(all_aligned_sentences, output_file_path)
        elif output_format == "json":
            write_to_json(all_aligned_sentences, output_file_path)
        
        # Write detailed alignment report
        write_match_report(overall_stats, output_file_path)
        return True
    else:
        print("No sentences were aligned across any chapters. Output file not written.")
        return False


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

    success = process_books(
        english_book_path=args.english_book_path,
        chinese_book_path=args.chinese_book_path,
        output_file_path=args.output_file_path,
        output_format=args.format,
        similarity_threshold=args.similarity_threshold,
        manual_align=args.align,
        auto_align=args.auto_align,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 