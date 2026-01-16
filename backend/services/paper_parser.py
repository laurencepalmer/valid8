import os
import fitz  # PyMuPDF
from backend.models.paper import Paper, PaperSection


def parse_pdf(file_path: str) -> Paper:
    doc = fitz.open(file_path)
    full_text = ""
    sections = []
    current_offset = 0

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            section = PaperSection(
                text=page_text,
                start_idx=current_offset,
                end_idx=current_offset + len(page_text),
                page=page_num + 1,
            )
            sections.append(section)
            full_text += page_text
            current_offset += len(page_text)

    name = os.path.basename(file_path)
    doc.close()

    return Paper(
        name=name,
        content=full_text,
        sections=sections,
        source_type="pdf",
        page_count=len(sections),
        file_path=file_path,
    )


def parse_text(content: str, name: str = "document.txt") -> Paper:
    paragraphs = content.split("\n\n")
    sections = []
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
    import re

    header_pattern = re.compile(
        r"^(#{1,6}\s+.+|(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References|Acknowledgments?))\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    sections = []
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
            sections.append(
                PaperSection(
                    text=section_text,
                    start_idx=start,
                    end_idx=end,
                )
            )

    return sections


def chunk_paper(paper: Paper, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Chunk paper content for embedding and indexing.

    Strategy:
    - For PDFs: Use page sections, then chunk within pages if too large
    - For text: Chunk by character count with overlap
    - Maintain overlap for context continuity

    Returns list of dicts with: content, start_idx, end_idx, page
    """
    chunks = []

    if paper.source_type == "pdf" and paper.sections:
        # Use page-based sections from PDF
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
    else:
        # Text-based chunking
        content = paper.content
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
