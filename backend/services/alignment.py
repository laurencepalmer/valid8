import json
from typing import Optional

from backend.models.analysis import (
    CodeReference,
    HighlightAnalysisResponse,
    AlignmentCheckResponse,
    AlignmentIssue,
)
from backend.models.codebase import Codebase
from backend.services.ai import get_ai_provider
from backend.services.embeddings import get_embedding_service
from backend.services.state import app_state


HIGHLIGHT_SYSTEM_PROMPT = """You are an expert at analyzing research papers and their associated code implementations.
Given a highlighted section from a research paper and relevant code snippets, your task is to:
1. Identify which code snippets are most relevant to the highlighted text
2. Explain how each code snippet relates to the paper content
3. Provide a brief summary of the connection

Respond in JSON format with the following structure:
{
    "relevant_snippets": [
        {
            "index": <index of the snippet from the provided list>,
            "relevance_score": <0.0 to 1.0>,
            "explanation": "<how this code relates to the highlighted text>"
        }
    ],
    "summary": "<brief overall summary of the paper-to-code connection>"
}"""


ALIGNMENT_SYSTEM_PROMPT = """You are an expert code reviewer analyzing whether code matches its described functionality.
Given a user's summary of what code should do and the actual code, your task is to:
1. Evaluate how well the code matches the description
2. Identify any discrepancies (missing features, incorrect implementations, extra features)
3. Provide an alignment score and suggestions

Respond in JSON format with the following structure:
{
    "alignment_score": <0.0 to 1.0>,
    "is_aligned": <true if score >= 0.7>,
    "issues": [
        {
            "issue_type": "<missing|incorrect|extra>",
            "description": "<description of the issue>",
            "summary_excerpt": "<relevant part of the summary, if applicable>",
            "code_location": "<file:line if applicable>"
        }
    ],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"],
    "summary": "<overall assessment>"
}"""


async def analyze_highlight(
    highlighted_text: str,
    codebase: Optional[Codebase] = None,
    n_results: int = 5,
) -> HighlightAnalysisResponse:
    if codebase is None:
        codebase = app_state.codebase

    if codebase is None:
        raise ValueError("No codebase loaded")

    embedding_service = get_embedding_service()
    similar_chunks = await embedding_service.search_similar_async(
        highlighted_text, n_results=n_results
    )

    if not similar_chunks:
        return HighlightAnalysisResponse(
            highlighted_text=highlighted_text,
            code_references=[],
            summary="No relevant code found for the highlighted text.",
        )

    chunks_text = "\n\n".join(
        f"[Snippet {i}]\n{chunk['document']}" for i, chunk in enumerate(similar_chunks)
    )

    prompt = f"""Highlighted text from paper:
\"\"\"{highlighted_text}\"\"\"

Potentially relevant code snippets:
{chunks_text}

Analyze which snippets are most relevant to the highlighted text."""

    ai_provider = get_ai_provider()
    response = await ai_provider.complete(prompt, system_prompt=HIGHLIGHT_SYSTEM_PROMPT)

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx]
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {
            "relevant_snippets": [
                {"index": i, "relevance_score": chunk["similarity"], "explanation": ""}
                for i, chunk in enumerate(similar_chunks[:3])
            ],
            "summary": "Analysis completed using embedding similarity.",
        }

    code_references = []
    for snippet_info in result.get("relevant_snippets", []):
        idx = snippet_info.get("index", 0)
        if idx < len(similar_chunks):
            chunk = similar_chunks[idx]
            metadata = chunk["metadata"]

            lines = chunk["document"].split("\n")
            content_start = next(
                (i for i, line in enumerate(lines) if not line.startswith("File:")),
                0,
            )
            content = "\n".join(lines[content_start:])

            code_references.append(
                CodeReference(
                    file_path=metadata["file_path"],
                    relative_path=metadata["relative_path"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    content=content,
                    relevance_score=snippet_info.get(
                        "relevance_score", chunk["similarity"]
                    ),
                    explanation=snippet_info.get(
                        "explanation", "Matched via semantic similarity"
                    ),
                )
            )

    return HighlightAnalysisResponse(
        highlighted_text=highlighted_text,
        code_references=code_references,
        summary=result.get("summary", ""),
    )


async def check_alignment(
    summary: str,
    file_paths: Optional[list[str]] = None,
    codebase: Optional[Codebase] = None,
) -> AlignmentCheckResponse:
    if codebase is None:
        codebase = app_state.codebase

    if codebase is None:
        raise ValueError("No codebase loaded")

    if file_paths:
        relevant_files = [f for f in codebase.files if f.relative_path in file_paths]
    else:
        embedding_service = get_embedding_service()
        similar_chunks = await embedding_service.search_similar_async(
            summary, n_results=10
        )

        seen_files = set()
        relevant_files = []
        for chunk in similar_chunks:
            rel_path = chunk["metadata"]["relative_path"]
            if rel_path not in seen_files:
                seen_files.add(rel_path)
                for f in codebase.files:
                    if f.relative_path == rel_path:
                        relevant_files.append(f)
                        break

    if not relevant_files:
        return AlignmentCheckResponse(
            alignment_score=0.0,
            summary="No relevant code files found to compare against.",
            is_aligned=False,
            issues=[
                AlignmentIssue(
                    issue_type="missing",
                    description="Could not find any code matching the summary description",
                )
            ],
            suggestions=["Ensure the codebase is properly loaded and indexed"],
        )

    code_content = "\n\n".join(
        f"=== {f.relative_path} ===\n{f.content[:5000]}"
        for f in relevant_files[:5]
    )

    prompt = f"""User's summary of what the code should do:
\"\"\"{summary}\"\"\"

Actual code:
{code_content}

Evaluate how well the code matches the user's description."""

    ai_provider = get_ai_provider()
    response = await ai_provider.complete(prompt, system_prompt=ALIGNMENT_SYSTEM_PROMPT)

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx]
        result = json.loads(response)
    except json.JSONDecodeError:
        return AlignmentCheckResponse(
            alignment_score=0.5,
            summary="Could not parse AI response. Manual review recommended.",
            is_aligned=False,
            issues=[],
            suggestions=["Try rephrasing your summary for better analysis"],
        )

    issues = []
    for issue_data in result.get("issues", []):
        code_ref = None
        if "code_location" in issue_data and issue_data["code_location"]:
            loc = issue_data["code_location"]
            if ":" in loc:
                file_part, line_part = loc.rsplit(":", 1)
                try:
                    line_num = int(line_part)
                    code_ref = CodeReference(
                        file_path=file_part,
                        relative_path=file_part,
                        start_line=line_num,
                        end_line=line_num,
                        content="",
                        relevance_score=0.0,
                        explanation=issue_data.get("description", ""),
                    )
                except ValueError:
                    pass

        issues.append(
            AlignmentIssue(
                issue_type=issue_data.get("issue_type", "incorrect"),
                description=issue_data.get("description", ""),
                summary_excerpt=issue_data.get("summary_excerpt"),
                code_reference=code_ref,
            )
        )

    alignment_score = result.get("alignment_score", 0.5)

    return AlignmentCheckResponse(
        alignment_score=alignment_score,
        summary=result.get("summary", ""),
        is_aligned=result.get("is_aligned", alignment_score >= 0.7),
        issues=issues,
        suggestions=result.get("suggestions", []),
    )
