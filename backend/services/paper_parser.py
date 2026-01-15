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
