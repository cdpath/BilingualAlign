#!/usr/bin/env python3
"""
Preprocess English sample book to add markdown headers for chapter detection.
"""

import re

def preprocess_english_book(input_path, output_path):
    """Add markdown headers to English chapter titles."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if it's a standalone chapter heading
        if re.match(r'^PROLOGUE$', stripped):
            processed_lines.append(f"# {stripped}")
        elif re.match(r'^CHAPTER \d+$', stripped):
            processed_lines.append(f"# {stripped}")
        elif re.match(r'^EPILOGUE$', stripped):
            processed_lines.append(f"# {stripped}")
        else:
            processed_lines.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    print(f"Processed English file: {input_path} -> {output_path}")

if __name__ == "__main__":
    preprocess_english_book('sample_books/english.md', 'sample_books/english_processed.md')
    print("English preprocessing complete!") 