import os
import re
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import Optional

import fitz  # PyMuPDF

from backend.models.paper import Paper, PaperSection

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Structured Processing
# =============================================================================


class TextRole(Enum):
    TITLE = "title"
    SECTION_HEADER = "section_header"
    SUBSECTION_HEADER = "subsection_header"
    BODY = "body"


@dataclass
class TextSpan:
    """Individual text span with font information."""

    text: str
    size: float
    bold: bool
    italic: bool
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class TextBlock:
    """Raw block from PyMuPDF with lines/spans."""

    text: str
    spans: list[TextSpan]
    bbox: tuple[float, float, float, float]
    page_num: int
    dominant_size: float = 0.0
    is_bold: bool = False

    def __post_init__(self):
        if self.spans and not self.dominant_size:
            # Calculate dominant font size (weighted by text length)
            size_weights: dict[float, int] = defaultdict(int)
            bold_chars = 0
            total_chars = 0
            for span in self.spans:
                size_weights[span.size] += len(span.text)
                total_chars += len(span.text)
                if span.bold:
                    bold_chars += len(span.text)

            if size_weights:
                self.dominant_size = max(size_weights, key=lambda s: size_weights[s])
            if total_chars > 0:
                self.is_bold = bold_chars > total_chars * 0.5


@dataclass
class ProcessedBlock:
    """Classified block with semantic role."""

    block: TextBlock
    role: TextRole
    section_name: Optional[str] = None


@dataclass
class FontProfile:
    """Document font size analysis results."""

    title_size: float
    section_header_size: float
    body_size: float
    size_tolerance: float = 0.5


@dataclass
class SemanticSection:
    """Detected section with name, content, and page range."""

    name: str
    section_type: str  # "abstract", "body", "header"
    content: str
    start_page: int
    end_page: int
    start_idx: int = 0
    end_idx: int = 0
    blocks: list[ProcessedBlock] = field(default_factory=list)


# =============================================================================
# Section Detection Patterns
# =============================================================================

SECTION_PATTERNS = [
    # Numbered sections with common names
    (
        r"^\s*\d+\.?\s+(introduction|methods?|methodology|results?|discussion|conclusions?|background|related\s+work|experiments?|analysis|implementation|evaluation|limitations?|future\s+work)\s*$",
        TextRole.SECTION_HEADER,
    ),
    # Unnumbered common section headers
    (
        r"^\s*(abstract|introduction|methods?|methodology|results?|discussion|conclusions?|background|related\s+work|experiments?|analysis|acknowledgments?|declarations?|data\s+availability|supplementary|appendix)\s*$",
        TextRole.SECTION_HEADER,
    ),
    # Numbered subsections (e.g., "2.1 Data Collection")
    (r"^\s*\d+\.\d+\.?\s+\w", TextRole.SUBSECTION_HEADER),
    # Lettered subsections (e.g., "A. Overview")
    (r"^\s*[A-Z]\.\s+\w", TextRole.SUBSECTION_HEADER),
]

REFERENCES_PATTERNS = [
    r"^\s*references?\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works?\s+cited\s*$",
    r"^\s*literature\s+cited\s*$",
    r"^\s*\d+\.?\s+references?\s*$",
]


# =============================================================================
# Extraction Layer
# =============================================================================


def extract_structured_blocks(doc: fitz.Document) -> list[TextBlock]:
    """Extract blocks with font size, bbox, bold flags from dict structure."""
    all_blocks: list[TextBlock] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_blocks = page_dict.get("blocks", [])

        for block in page_blocks:
            # Skip image blocks
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            if not lines:
                continue

            spans_list: list[TextSpan] = []
            block_text_parts: list[str] = []

            for line in lines:
                line_spans = line.get("spans", [])
                line_texts: list[str] = []

                for span in line_spans:
                    span_text = span.get("text", "")
                    if not span_text:
                        continue

                    # Extract font flags: bit 4 = bold, bit 1 = italic
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & (1 << 4)) or bool(flags & (1 << 5))
                    is_italic = bool(flags & (1 << 1))

                    spans_list.append(
                        TextSpan(
                            text=span_text,
                            size=span.get("size", 0),
                            bold=is_bold,
                            italic=is_italic,
                            bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        )
                    )
                    line_texts.append(span_text)

                if line_texts:
                    block_text_parts.append("".join(line_texts))

            if spans_list:
                block_text = "\n".join(block_text_parts)
                all_blocks.append(
                    TextBlock(
                        text=block_text,
                        spans=spans_list,
                        bbox=tuple(block.get("bbox", (0, 0, 0, 0))),
                        page_num=page_num,
                    )
                )

    return all_blocks


# =============================================================================
# Header/Footer Filtering
# =============================================================================


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for header/footer comparison (replace numbers with #)."""
    normalized = re.sub(r"\d+", "#", text)
    normalized = normalized.strip().lower()
    return normalized


def detect_repeated_blocks(
    blocks: list[TextBlock], total_pages: int, threshold: float = 0.7
) -> set[str]:
    """Find text appearing on >70% of pages with similar bbox in top/bottom 10%."""
    if total_pages < 3:
        return set()

    # Group blocks by normalized text
    text_occurrences: dict[str, list[TextBlock]] = defaultdict(list)

    for block in blocks:
        normalized = normalize_text_for_comparison(block.text)
        if normalized and len(normalized) < 100:  # Headers/footers are typically short
            text_occurrences[normalized].append(block)

    repeated_patterns: set[str] = set()

    for normalized_text, occurrences in text_occurrences.items():
        # Check if appears on enough pages
        unique_pages = set(b.page_num for b in occurrences)
        if len(unique_pages) < total_pages * threshold:
            continue

        # Check if y-position is consistent (within 10pt)
        y_positions = [b.bbox[1] for b in occurrences]
        y_variance = max(y_positions) - min(y_positions)
        if y_variance > 20:
            continue

        repeated_patterns.add(normalized_text)

    return repeated_patterns


def filter_headers_footers(
    blocks: list[TextBlock], repeated_patterns: set[str], page_height: float = 800
) -> list[TextBlock]:
    """Remove blocks matching header/footer patterns."""
    filtered: list[TextBlock] = []

    for block in blocks:
        normalized = normalize_text_for_comparison(block.text)

        # Check if matches repeated pattern
        if normalized in repeated_patterns:
            # Verify it's in header/footer region (top/bottom 10%)
            y_pos = block.bbox[1]
            y_end = block.bbox[3]
            in_header = y_pos < page_height * 0.1
            in_footer = y_end > page_height * 0.9

            if in_header or in_footer:
                logger.debug(f"Filtered header/footer: {block.text[:50]}")
                continue

        filtered.append(block)

    return filtered


# =============================================================================
# References Section Filtering
# =============================================================================


def find_references_start(
    blocks: list[ProcessedBlock],
) -> Optional[int]:
    """Find block index where references section starts."""
    for i, pb in enumerate(blocks):
        text = pb.block.text.strip()

        for pattern in REFERENCES_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                # Verify it looks like a header (larger font or bold)
                if (
                    pb.role
                    in (TextRole.SECTION_HEADER, TextRole.SUBSECTION_HEADER, TextRole.TITLE)
                    or pb.block.is_bold
                ):
                    logger.debug(f"Found references at block {i}: {text}")
                    return i

    return None


def filter_references_section(blocks: list[ProcessedBlock]) -> list[ProcessedBlock]:
    """Return blocks before references section."""
    ref_start = find_references_start(blocks)
    if ref_start is not None:
        return blocks[:ref_start]
    return blocks


# =============================================================================
# Two-Column Layout Handling
# =============================================================================


def detect_column_layout(page_blocks: list[TextBlock], page_width: float) -> bool:
    """Check if blocks cluster in left/right halves with gap in middle."""
    if len(page_blocks) < 4:
        return False

    # Get x-center of each block
    x_centers = [(b.bbox[0] + b.bbox[2]) / 2 for b in page_blocks]

    left_count = sum(1 for x in x_centers if x < page_width * 0.4)
    right_count = sum(1 for x in x_centers if x > page_width * 0.6)
    middle_count = sum(1 for x in x_centers if page_width * 0.4 <= x <= page_width * 0.6)

    # Two-column if most blocks are in left/right thirds with few in middle
    total = len(x_centers)
    is_two_column = (
        left_count > total * 0.3
        and right_count > total * 0.3
        and middle_count < total * 0.15
    )

    return is_two_column


def reorder_two_column_blocks(
    blocks: list[TextBlock], page_width: float
) -> list[TextBlock]:
    """Sort: left column (top-to-bottom), then right column (top-to-bottom)."""
    midpoint = page_width / 2

    left_col: list[TextBlock] = []
    right_col: list[TextBlock] = []

    for block in blocks:
        x_center = (block.bbox[0] + block.bbox[2]) / 2
        if x_center < midpoint:
            left_col.append(block)
        else:
            right_col.append(block)

    # Sort each column by y-position (top to bottom)
    left_col.sort(key=lambda b: b.bbox[1])
    right_col.sort(key=lambda b: b.bbox[1])

    return left_col + right_col


def process_page_layout(
    page_blocks: list[TextBlock], page_width: float
) -> list[TextBlock]:
    """Detect layout and reorder appropriately per page."""
    if detect_column_layout(page_blocks, page_width):
        logger.debug(f"Detected two-column layout on page {page_blocks[0].page_num if page_blocks else '?'}")
        return reorder_two_column_blocks(page_blocks, page_width)
    else:
        # Single column: sort by y-position only
        return sorted(page_blocks, key=lambda b: b.bbox[1])


# =============================================================================
# Font Size Histogram Analysis
# =============================================================================


def build_font_profile(
    blocks: list[TextBlock], size_tolerance: float = 0.5
) -> FontProfile:
    """Analyze font sizes to identify title, headers, and body text."""
    # Collect font sizes with character counts
    size_char_counts: dict[float, int] = defaultdict(int)

    for block in blocks:
        for span in block.spans:
            # Round to nearest 0.5pt for clustering
            rounded_size = round(span.size * 2) / 2
            size_char_counts[rounded_size] += len(span.text)

    if not size_char_counts:
        return FontProfile(
            title_size=14.0, section_header_size=12.0, body_size=10.0
        )

    # Sort by character count (most frequent first)
    sorted_sizes = sorted(size_char_counts.items(), key=lambda x: -x[1])

    # Body text is the most frequent size
    body_size = sorted_sizes[0][0]

    # Find sizes larger than body
    larger_sizes = sorted(
        [s for s, _ in sorted_sizes if s > body_size + size_tolerance], reverse=True
    )

    # Title is the largest (usually rare, on first page)
    title_size = larger_sizes[0] if larger_sizes else body_size + 4

    # Section header is second largest or slightly larger than body
    if len(larger_sizes) >= 2:
        section_header_size = larger_sizes[1]
    elif larger_sizes:
        section_header_size = larger_sizes[0]
    else:
        section_header_size = body_size + 2

    logger.debug(
        f"Font profile: title={title_size}, header={section_header_size}, body={body_size}"
    )

    return FontProfile(
        title_size=title_size,
        section_header_size=section_header_size,
        body_size=body_size,
        size_tolerance=size_tolerance,
    )


# =============================================================================
# Section Detection (Font + Regex)
# =============================================================================


def classify_block_role(
    block: TextBlock, font_profile: FontProfile
) -> tuple[TextRole, Optional[str]]:
    """Classify block as title, header, or body based on font and patterns."""
    text = block.text.strip()

    if not text:
        return TextRole.BODY, None

    # First check regex patterns for explicit section names
    for pattern, role in SECTION_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            # Extract section name from match
            section_name = text.strip()
            return role, section_name

    # Check font size against profile thresholds
    size = block.dominant_size
    tolerance = font_profile.size_tolerance

    # Title: largest font, usually first page, short text
    if size >= font_profile.title_size - tolerance and len(text) < 200:
        if block.page_num == 0:  # Titles typically on first page
            return TextRole.TITLE, text

    # Section header: larger than body, often bold, short text
    if size >= font_profile.section_header_size - tolerance:
        if len(text) < 100 and (block.is_bold or size > font_profile.body_size + 1):
            return TextRole.SECTION_HEADER, text

    # Default to body
    return TextRole.BODY, None


def process_blocks_with_roles(
    blocks: list[TextBlock], font_profile: FontProfile
) -> list[ProcessedBlock]:
    """Classify all blocks with their semantic roles."""
    processed: list[ProcessedBlock] = []

    for block in blocks:
        role, section_name = classify_block_role(block, font_profile)
        processed.append(ProcessedBlock(block=block, role=role, section_name=section_name))

    return processed


# =============================================================================
# Abstract Extraction
# =============================================================================


def extract_abstract(
    blocks: list[ProcessedBlock],
) -> tuple[Optional[SemanticSection], list[ProcessedBlock]]:
    """Find and extract abstract section."""
    abstract_start: Optional[int] = None
    abstract_end: Optional[int] = None

    for i, pb in enumerate(blocks):
        text_lower = pb.block.text.strip().lower()

        # Find "Abstract" header
        if abstract_start is None:
            if text_lower == "abstract" or re.match(r"^\s*abstract\s*[:.]?\s*$", text_lower):
                abstract_start = i
                continue

        # If we found abstract, look for next section header
        if abstract_start is not None and abstract_end is None:
            if pb.role in (TextRole.SECTION_HEADER, TextRole.SUBSECTION_HEADER):
                if i > abstract_start:  # Don't immediately end
                    abstract_end = i
                    break

    if abstract_start is None:
        return None, blocks

    if abstract_end is None:
        # Abstract might extend to a reasonable point
        # Look for the next header within first ~10 blocks
        abstract_end = min(abstract_start + 10, len(blocks))
        for i in range(abstract_start + 1, min(abstract_start + 15, len(blocks))):
            if i < len(blocks) and blocks[i].role in (
                TextRole.SECTION_HEADER,
                TextRole.SUBSECTION_HEADER,
            ):
                abstract_end = i
                break

    # Build abstract section
    abstract_blocks = blocks[abstract_start:abstract_end]
    abstract_content = "\n".join(pb.block.text for pb in abstract_blocks).strip()

    # Remove "Abstract" header from content if it's just the word
    if abstract_content.lower().startswith("abstract"):
        lines = abstract_content.split("\n")
        if lines and lines[0].strip().lower() in ("abstract", "abstract:"):
            abstract_content = "\n".join(lines[1:]).strip()

    if not abstract_content:
        return None, blocks

    abstract_section = SemanticSection(
        name="Abstract",
        section_type="abstract",
        content=abstract_content,
        start_page=abstract_blocks[0].block.page_num + 1 if abstract_blocks else 1,
        end_page=abstract_blocks[-1].block.page_num + 1 if abstract_blocks else 1,
        blocks=abstract_blocks,
    )

    # Return remaining blocks (before abstract + after abstract)
    remaining = blocks[:abstract_start] + blocks[abstract_end:]

    return abstract_section, remaining


# =============================================================================
# Semantic Section Building
# =============================================================================


def build_semantic_sections(
    blocks: list[ProcessedBlock], abstract: Optional[SemanticSection] = None
) -> list[SemanticSection]:
    """Group blocks by section headers into semantic sections."""
    sections: list[SemanticSection] = []

    if abstract:
        sections.append(abstract)

    current_section: Optional[SemanticSection] = None
    current_content_parts: list[str] = []
    current_blocks: list[ProcessedBlock] = []

    def finalize_section():
        nonlocal current_section, current_content_parts, current_blocks
        if current_section:
            current_section.content = "\n".join(current_content_parts).strip()
            current_section.blocks = current_blocks.copy()
            if current_blocks:
                current_section.end_page = current_blocks[-1].block.page_num + 1
            if current_section.content:
                sections.append(current_section)
        current_content_parts = []
        current_blocks = []

    for pb in blocks:
        if pb.role in (TextRole.SECTION_HEADER, TextRole.TITLE):
            # Finalize previous section
            finalize_section()

            # Start new section
            current_section = SemanticSection(
                name=pb.section_name or pb.block.text.strip(),
                section_type="body",
                content="",
                start_page=pb.block.page_num + 1,
                end_page=pb.block.page_num + 1,
            )
            # Don't add the header text to content
            current_blocks = [pb]

        elif pb.role == TextRole.SUBSECTION_HEADER:
            # Add subsection as part of current section content
            if current_section:
                current_content_parts.append(f"\n{pb.block.text.strip()}\n")
                current_blocks.append(pb)
            else:
                # No section yet, start one
                current_section = SemanticSection(
                    name=pb.section_name or pb.block.text.strip(),
                    section_type="body",
                    content="",
                    start_page=pb.block.page_num + 1,
                    end_page=pb.block.page_num + 1,
                )
                current_blocks = [pb]

        else:  # BODY
            if current_section:
                current_content_parts.append(pb.block.text)
                current_blocks.append(pb)
            else:
                # Body text before any section header
                # Create an implicit "Introduction" or "Body" section
                current_section = SemanticSection(
                    name="Body",
                    section_type="body",
                    content="",
                    start_page=pb.block.page_num + 1,
                    end_page=pb.block.page_num + 1,
                )
                current_content_parts.append(pb.block.text)
                current_blocks = [pb]

    # Finalize last section
    finalize_section()

    return sections


# =============================================================================
# Token-Aware Chunking
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic (~85% accurate for English)."""
    return len(text) // 4


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_section(section: SemanticSection, max_tokens: int = 800) -> list[dict]:
    """Chunk a section into pieces respecting token limits."""
    content = section.content
    if not content:
        return []

    content_tokens = estimate_tokens(content)

    # If section fits in one chunk, return as-is
    if content_tokens <= max_tokens:
        return [
            {
                "content": content,
                "start_idx": section.start_idx,
                "end_idx": section.end_idx,
                "page": section.start_page,
                "section_name": section.name,
                "section_type": section.section_type,
            }
        ]

    # Split by paragraphs first
    paragraphs = split_into_paragraphs(content)
    chunks: list[dict] = []
    current_chunk_parts: list[str] = []
    current_tokens = 0
    chunk_start_idx = section.start_idx
    overlap_tokens = 50
    overlap_text = ""

    def finalize_chunk():
        nonlocal current_chunk_parts, current_tokens, chunk_start_idx, overlap_text
        if current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts)
            # Prepend overlap from previous chunk
            if overlap_text and chunks:
                chunk_text = overlap_text + "\n\n" + chunk_text

            chunk_end_idx = chunk_start_idx + len(chunk_text)
            chunks.append(
                {
                    "content": chunk_text.strip(),
                    "start_idx": chunk_start_idx,
                    "end_idx": chunk_end_idx,
                    "page": section.start_page,
                    "section_name": section.name,
                    "section_type": section.section_type,
                }
            )
            # Set up overlap for next chunk
            overlap_text = current_chunk_parts[-1][:overlap_tokens * 4] if current_chunk_parts else ""
            chunk_start_idx = chunk_end_idx
            current_chunk_parts = []
            current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If single paragraph exceeds limit, split by sentences
        if para_tokens > max_tokens:
            finalize_chunk()  # Finish current chunk
            sentences = split_into_sentences(para)
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_chunk_parts:
                    finalize_chunk()
                current_chunk_parts.append(sent)
                current_tokens += sent_tokens
        elif current_tokens + para_tokens > max_tokens:
            finalize_chunk()
            current_chunk_parts.append(para)
            current_tokens = para_tokens
        else:
            current_chunk_parts.append(para)
            current_tokens += para_tokens

    finalize_chunk()

    return chunks


# =============================================================================
# Main Parsing Functions
# =============================================================================


def parse_pdf(file_path: str) -> Paper:
    """Parse PDF with structured text extraction and semantic section detection."""
    doc = fitz.open(file_path)
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        return Paper(
            name=os.path.basename(file_path),
            content="",
            sections=[],
            source_type="pdf",
            page_count=0,
            file_path=file_path,
        )

    # Get page dimensions from first page
    first_page = doc[0]
    page_width = first_page.rect.width
    page_height = first_page.rect.height

    # Step 1: Extract structured blocks
    all_blocks = extract_structured_blocks(doc)

    if not all_blocks:
        # Possibly scanned PDF without text
        logger.warning(f"No text extracted from PDF: {file_path}")
        doc.close()
        return Paper(
            name=os.path.basename(file_path),
            content="",
            sections=[],
            source_type="pdf",
            page_count=total_pages,
            file_path=file_path,
        )

    # Step 2: Detect and filter headers/footers
    repeated_patterns = detect_repeated_blocks(all_blocks, total_pages)
    filtered_blocks = filter_headers_footers(all_blocks, repeated_patterns, page_height)

    # Step 3: Process page layout (handle two-column)
    # Group blocks by page
    blocks_by_page: dict[int, list[TextBlock]] = defaultdict(list)
    for block in filtered_blocks:
        blocks_by_page[block.page_num].append(block)

    # Reorder each page
    ordered_blocks: list[TextBlock] = []
    for page_num in sorted(blocks_by_page.keys()):
        page_blocks = blocks_by_page[page_num]
        ordered_page_blocks = process_page_layout(page_blocks, page_width)
        ordered_blocks.extend(ordered_page_blocks)

    # Step 4: Build font profile
    font_profile = build_font_profile(ordered_blocks)

    # Step 5: Classify blocks with roles
    processed_blocks = process_blocks_with_roles(ordered_blocks, font_profile)

    # Step 6: Filter references section
    processed_blocks = filter_references_section(processed_blocks)

    # Step 7: Extract abstract
    abstract, remaining_blocks = extract_abstract(processed_blocks)

    # Step 8: Build semantic sections
    sections = build_semantic_sections(remaining_blocks, abstract)

    # Build full text and PaperSection objects
    full_text_parts: list[str] = []
    paper_sections: list[PaperSection] = []
    current_offset = 0

    for section in sections:
        section_text = section.content
        if not section_text:
            continue

        full_text_parts.append(section_text)

        paper_section = PaperSection(
            text=section_text,
            start_idx=current_offset,
            end_idx=current_offset + len(section_text),
            page=section.start_page,
            section_name=section.name,
            section_type=section.section_type,
        )
        paper_sections.append(paper_section)
        current_offset += len(section_text) + 1  # +1 for join separator

    full_text = "\n".join(full_text_parts)

    doc.close()

    return Paper(
        name=os.path.basename(file_path),
        content=full_text,
        sections=paper_sections,
        source_type="pdf",
        page_count=total_pages,
        file_path=file_path,
    )


def parse_text(content: str, name: str = "document.txt") -> Paper:
    """Parse plain text content into Paper structure."""
    paragraphs = content.split("\n\n")
    sections: list[PaperSection] = []
    current_offset = 0

    for para in paragraphs:
        if para.strip():
            section = PaperSection(
                text=para,
                start_idx=current_offset,
                end_idx=current_offset + len(para),
            )
            sections.append(section)
        current_offset += len(para) + 2  # +2 for the \n\n

    return Paper(
        name=name,
        content=content,
        sections=sections,
        source_type="text",
    )


def extract_sections_by_headers(content: str) -> list[PaperSection]:
    """Extract sections from text by looking for header patterns."""
    header_pattern = re.compile(
        r"^(#{1,6}\s+.+|(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References|Acknowledgments?))\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    sections: list[PaperSection] = []
    matches = list(header_pattern.finditer(content))

    if not matches:
        return [
            PaperSection(
                text=content,
                start_idx=0,
                end_idx=len(content),
            )
        ]

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()

        if section_text:
            # Extract section name from header
            header_text = match.group(0).strip()
            section_name = re.sub(r"^#+\s*", "", header_text)

            sections.append(
                PaperSection(
                    text=section_text,
                    start_idx=start,
                    end_idx=end,
                    section_name=section_name,
                )
            )

    return sections


def chunk_paper(
    paper: Paper, chunk_size: int = 500, overlap: int = 100, max_tokens: int = 800
) -> list[dict]:
    """
    Chunk paper content for embedding and indexing.

    Strategy:
    - For PDFs with semantic sections: chunk primarily by section
    - For PDFs without sections or text: character-based chunking
    - Maintain overlap for context continuity

    Returns list of dicts with: content, start_idx, end_idx, page
    """
    chunks: list[dict] = []

    if paper.source_type == "pdf" and paper.sections:
        # Check if we have semantic sections (with section_name)
        has_semantic_sections = any(s.section_name for s in paper.sections)

        if has_semantic_sections:
            # Use section-aware chunking
            for section in paper.sections:
                if not section.text.strip():
                    continue

                # Create a SemanticSection for chunking
                semantic_section = SemanticSection(
                    name=section.section_name or "Body",
                    section_type=section.section_type or "body",
                    content=section.text,
                    start_page=section.page or 1,
                    end_page=section.page or 1,
                    start_idx=section.start_idx,
                    end_idx=section.end_idx,
                )

                section_chunks = chunk_section(semantic_section, max_tokens)
                chunks.extend(section_chunks)
        else:
            # Fall back to page-based chunking for PDFs without semantic sections
            chunks = _chunk_by_pages(paper, chunk_size, overlap)
    else:
        # Text-based chunking
        chunks = _chunk_text_content(paper.content, chunk_size, overlap)

    return chunks


def _chunk_by_pages(paper: Paper, chunk_size: int, overlap: int) -> list[dict]:
    """Chunk PDF by pages when semantic sections aren't available."""
    chunks: list[dict] = []

    for section in paper.sections:
        page_text = section.text.strip()
        page_num = section.page

        if not page_text:
            continue

        if len(page_text) <= chunk_size:
            chunks.append(
                {
                    "content": page_text,
                    "start_idx": section.start_idx,
                    "end_idx": section.end_idx,
                    "page": page_num,
                }
            )
        else:
            # Split long pages into smaller chunks
            start = 0
            while start < len(page_text):
                end = min(start + chunk_size, len(page_text))

                # Try to break at sentence boundary
                if end < len(page_text):
                    last_period = page_text.rfind(".", start, end)
                    last_newline = page_text.rfind("\n", start, end)
                    break_point = max(last_period, last_newline)
                    if break_point > start + chunk_size // 2:
                        end = break_point + 1

                chunk_text = page_text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            "content": chunk_text,
                            "start_idx": section.start_idx + start,
                            "end_idx": section.start_idx + end,
                            "page": page_num,
                        }
                    )

                start = end - overlap if end < len(page_text) else end

    return chunks


def _chunk_text_content(content: str, chunk_size: int, overlap: int) -> list[dict]:
    """Chunk plain text content by character count with overlap."""
    chunks: list[dict] = []

    if not content:
        return chunks

    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to break at paragraph or sentence boundary
        if end < len(content):
            last_para = content.rfind("\n\n", start, end)
            last_period = content.rfind(".", start, end)
            break_point = max(last_para, last_period)
            if break_point > start + chunk_size // 2:
                end = break_point + 1

        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "content": chunk_text,
                    "start_idx": start,
                    "end_idx": end,
                    "page": None,
                }
            )

        start = end - overlap if end < len(content) else end

    return chunks
