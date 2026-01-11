#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Book Chunking using LangChain

Supports:
- Single chapter file: python chunk_book.py chapter.txt
- Full book with markers: python chunk_book.py book.txt --full-book

Chapter markers in full book: ### Chapter Title

Example book.txt:
    ### Про первое правило
    
    Им только дай волю…
    Первая учительница
    
    Совет «делать только то...
    
    ### Про смысл жизни
    
    Тот помер, не найдя смысла...
"""

import json
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).parent

CHUNK_SIZE = 1200      # Target chunk size in characters
CHUNK_OVERLAP = 0      # No overlap
MIN_CHUNK_SIZE = 400   # Merge chunks smaller than this

CHAPTER_MARKER = "### "  # Marker for chapter splits in full book

# Separators: section break first, then paragraph, then line, then sentence
SEPARATORS = ["\n\n…\n\n", "\n…\n", "…", "\n\n", "\n", ". ", ", ", " "]

# ============================================================
# TEXT PREPROCESSING
# ============================================================

def extract_dialogues(text: str, book_id: str, chapter_id: str = "") -> tuple[str, list]:
    """
    Extract ALL Q&A dialogues from chapter.
    
    Pattern - dialogues separated by …:
        Main content...
        
        …
        
        – Question/context from user
        – Answer from Labkovsky
    
    Returns:
        (main_text_without_dialogues, list of dialogue_dicts)
    """
    
    dialogues = []
    
    # Split by … separator
    parts = text.split('…')
    
    main_parts = []
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        if not part:
            continue
        
        # Check if this part starts with dialogue marker –
        lines = [l.strip() for l in part.split('\n') if l.strip()]
        
        if lines and lines[0].startswith('–'):
            # This is a dialogue block
            turns = []
            
            for line in lines:
                if line.startswith('–'):
                    content = line[1:].strip()  # Remove –
                    turns.append(content)
            
            if len(turns) >= 2:
                # Valid dialogue (at least question + answer)
                dialogue = {
                    "type": "dialogue",
                    "dialogue_id": len(dialogues),
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "turns": []
                }
                
                for j, turn in enumerate(turns):
                    role = "user" if j % 2 == 0 else "assistant"
                    dialogue["turns"].append({"role": role, "content": turn})
                
                dialogues.append(dialogue)
        else:
            # Regular content - keep it
            main_parts.append(part)
    
    # Reconstruct main text
    main_text = '\n\n'.join(main_parts)
    
    return main_text, dialogues


def remove_epigraphs(text: str) -> tuple[str, str]:
    """
    Remove epigraphs from book chapters.
    
    Returns:
        (cleaned_text, removed_content)
    """
    lines = text.split('\n')
    
    # Find where actual content starts
    content_start = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Skip short lines (likely title, epigraph, author)
        if len(line_stripped) < 50:
            continue
        
        # Found first substantial line - this is content
        content_start = i
        break
    
    result = '\n'.join(lines[content_start:])
    removed = '\n'.join(lines[:content_start])
    
    return result, removed


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split full book into chapters by marker.
    
    Returns:
        List of (chapter_title, chapter_content)
    """
    chapters = []
    
    parts = text.split(CHAPTER_MARKER)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # First line is title, rest is content
        lines = part.split('\n', 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        
        if title and content:
            chapters.append((title, content))
    
    return chapters


def chunk_text(text: str) -> list[str]:
    """Split text into chunks using LangChain splitter."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    
    # Merge small chunks with previous
    merged_chunks = []
    for chunk in chunks:
        if merged_chunks and len(chunk) < MIN_CHUNK_SIZE:
            merged_chunks[-1] = merged_chunks[-1] + "\n\n" + chunk
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks


def process_chapter(title: str, content: str, book_id: str, chunk_id_start: int = 0) -> tuple[list, list]:
    """
    Process a single chapter: extract dialogues, remove epigraphs, chunk.
    
    Returns:
        (list of chunk_dicts, list of dialogue_dicts)
    """
    print(f"\n{'='*60}")
    print(f"📖 Chapter: {title}")
    print(f"{'='*60}")
    
    # Convert title to safe filename-like id
    chapter_id = title.lower().replace(' ', '_').replace('про_', '')
    
    # Extract dialogues
    content, dialogues = extract_dialogues(content, book_id, chapter_id)
    if dialogues:
        print(f"📝 Extracted {len(dialogues)} dialogues")
    
    # Remove epigraphs
    content, removed = remove_epigraphs(content)
    if removed.strip():
        print(f"📝 Removed epigraph/header:")
        for line in removed.split('\n'):
            if line.strip():
                print(f"   | {line.strip()[:60]}")
    
    # Chunk
    text_chunks = chunk_text(content)
    print(f"📝 Created {len(text_chunks)} chunks")
    
    # Format results
    chunks = []
    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "chunk_id": chunk_id_start + i,
            "source": "book",
            "book_id": book_id,
            "chapter_id": chapter_id,
            "text": chunk,
            "char_count": len(chunk)
        })
        print(f"   Chunk {chunk_id_start + i}: {len(chunk)} chars - \"{chunk[:50]}...\"")
    
    return chunks, dialogues


# ============================================================
# MAIN
# ============================================================

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single chapter: python chunk_book.py chapter.txt")
        print("  Full book:      python chunk_book.py book.txt --full-book")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    full_book_mode = "--full-book" in sys.argv
    
    # Output files
    output_chunks = SCRIPT_DIR / f"{input_file.stem}_chunks.jsonl"
    output_dialogues = SCRIPT_DIR / f"{input_file.stem}_dialogues.jsonl"
    
    print("=" * 60)
    print("Book Chunking (LangChain)")
    print(f"Mode: {'Full Book' if full_book_mode else 'Single Chapter'}")
    print("=" * 60)
    
    # Load text
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📄 Input: {input_file.name}")
    print(f"📝 Total length: {len(text)} chars")
    
    all_chunks = []
    all_dialogues = []
    chunk_id = 0
    book_id = input_file.stem.lower()  # Use filename as book_id
    
    if full_book_mode:
        # Split into chapters
        chapters = split_into_chapters(text)
        print(f"📚 Found {len(chapters)} chapters")
        
        for title, content in chapters:
            chunks, dialogues = process_chapter(title, content, book_id, chunk_id)
            all_chunks.extend(chunks)
            all_dialogues.extend(dialogues)
            chunk_id += len(chunks)
    else:
        # Single chapter mode (original behavior)
        chunks, dialogues = process_chapter(input_file.stem, text, book_id, 0)
        all_chunks.extend(chunks)
        all_dialogues.extend(dialogues)
    
    # Save chunks
    output_chunks.parent.mkdir(parents=True, exist_ok=True)
    with open(output_chunks, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Save dialogues
    if all_dialogues:
        with open(output_dialogues, 'w', encoding='utf-8') as f:
            for d in all_dialogues:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    
    if full_book_mode:
        print(f"   Chapters: {len(chapters) if full_book_mode else 1}")
    
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Total dialogues: {len(all_dialogues)}")
    
    if all_chunks:
        sizes = [c["char_count"] for c in all_chunks]
        print(f"   Chunk sizes: {min(sizes)}-{max(sizes)} chars (avg {sum(sizes)//len(sizes)})")
    
    print(f"\n   Output chunks: {output_chunks}")
    if all_dialogues:
        print(f"   Output dialogues: {output_dialogues}")


if __name__ == "__main__":
    main()