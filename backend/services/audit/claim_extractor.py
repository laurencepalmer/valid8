"""Extract verifiable claims from description text."""

import asyncio
import json
import re
import uuid
from typing import Optional

from backend.models.audit import Claim, ClaimType
from backend.services.ai import get_ai_provider
from backend.services.logging import logger


# Sections to prioritize (contain verifiable implementation details)
IMPORTANT_SECTION_KEYWORDS = [
    "method", "methodology", "approach",
    "experiment", "experimental",
    "result", "results",
    "implementation", "implementation detail",
    "training", "training detail",
    "evaluation", "evaluation detail",
    "architecture", "model architecture", "network architecture",
    "dataset", "data",
    "setup", "experimental setup",
    "hyperparameter", "configuration", "config",
    "ablation",
]

# Sections to skip (unlikely to contain verifiable claims)
SKIP_SECTION_KEYWORDS = [
    "introduction",
    "related work", "related works", "prior work",
    "background",
    "conclusion", "conclusions",
    "future work", "future direction",
    "acknowledgment", "acknowledgement", "acknowledgments", "acknowledgements",
    "reference", "references", "bibliography",
    "abstract",
    "appendix",  # Often supplementary, can be added back if needed
]


# Common value patterns to extract and normalize
VALUE_PATTERNS = {
    "learning_rate": [
        r"learning\s*rate\s*(?:of|=|:)?\s*([0-9.e\-]+)",
        r"lr\s*(?:of|=|:)?\s*([0-9.e\-]+)",
    ],
    "batch_size": [
        r"batch\s*size\s*(?:of|=|:)?\s*(\d+)",
        r"mini-?batch(?:es)?\s*(?:of|=|:)?\s*(\d+)",
    ],
    "epochs": [
        r"(\d+)\s*epochs?",
        r"train(?:ed|ing)?\s*for\s*(\d+)\s*epochs?",
    ],
    "cv_folds": [
        r"(\d+)\s*-?\s*fold\s*(?:cross[- ]?validation|cv)",
        r"cross[- ]?validation\s*(?:with|of)?\s*(\d+)\s*folds?",
    ],
    "dropout": [
        r"dropout\s*(?:rate|probability)?\s*(?:of|=|:)?\s*([0-9.]+)",
    ],
    "weight_decay": [
        r"weight\s*decay\s*(?:of|=|:)?\s*([0-9.e\-]+)",
        r"l2\s*regularization\s*(?:of|=|:)?\s*([0-9.e\-]+)",
    ],
}


CLAIM_EXTRACTION_PROMPT = """You are analyzing a research paper or technical description to extract verifiable claims about the implementation.

Extract fine-grained, specific claims that can be verified by reading the code. Be precise about numbers, parameters, and specific details.

Good claims (fine-grained):
- "Uses 5-fold cross-validation"
- "Learning rate is 0.001"
- "Batch size is 32"
- "Uses Adam optimizer with beta1=0.9, beta2=0.999"
- "Data is split 80/10/10 for train/val/test"
- "Features are z-score normalized using training set statistics"
- "Uses ResNet-50 backbone pretrained on ImageNet"
- "Training runs for 100 epochs with early stopping patience of 10"

Bad claims (too vague):
- "Uses cross-validation" (how many folds?)
- "Uses a standard optimizer" (which one?)
- "Data is preprocessed" (how?)

Extract claims in these categories:
- data_preprocessing: How data is loaded, cleaned, transformed
- data_split: Train/val/test splits, cross-validation details
- model_architecture: Network structure, layers, sizes, pretrained weights
- training_procedure: Optimizer, learning rate, epochs, batch size, regularization
- evaluation_metrics: Specific metrics, how they're computed
- hyperparameter: Any specific hyperparameter values
- implementation_detail: Specific algorithms, libraries, techniques

For each claim, provide:
1. text: The exact or paraphrased claim
2. claim_type: One of the categories above
3. keywords: Key terms to search for in code (function names, variable names, values)
4. expected_value: The specific value claimed (if applicable)
5. source_location: Where in the text this was found (if identifiable)

Return a JSON array of claims.

Text to analyze:
{text}

Return ONLY valid JSON, no other text."""


class ClaimExtractor:
    """Extract verifiable claims from description text using LLM."""

    def __init__(self):
        self.ai_provider = None
        # Compile patterns for efficiency
        self._compiled_patterns = {
            key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in VALUE_PATTERNS.items()
        }

    async def extract_claims(
        self,
        text: str,
        focus_areas: Optional[list[ClaimType]] = None,
    ) -> list[Claim]:
        """
        Extract claims from description text.

        Args:
            text: The description text to analyze
            focus_areas: Optional filter to only extract certain claim types

        Returns:
            List of extracted claims
        """
        if self.ai_provider is None:
            self.ai_provider = get_ai_provider()

        all_claims = []

        # Step 1: Pattern-based extraction (fast, reliable for common values)
        pattern_claims = self._extract_pattern_claims(text)
        all_claims.extend(pattern_claims)
        logger.debug(f"Pattern extraction found {len(pattern_claims)} claims")

        # Step 2: LLM-based extraction (comprehensive)
        # For long texts, use multi-pass extraction
        if len(text) > 20000:
            llm_claims = await self._multi_pass_extraction(text)
        else:
            llm_claims = await self._single_pass_extraction(text)
        all_claims.extend(llm_claims)
        logger.debug(f"LLM extraction found {len(llm_claims)} claims")

        # Step 3: Deduplicate and merge
        all_claims = self._deduplicate_claims(all_claims)
        logger.info(f"Total unique claims: {len(all_claims)}")

        # Filter by focus areas if specified
        if focus_areas:
            all_claims = [c for c in all_claims if c.claim_type in focus_areas]

        return all_claims

    def _extract_pattern_claims(self, text: str) -> list[Claim]:
        """Extract claims using regex patterns for common values."""
        claims = []

        for value_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                for match in matches:
                    value = match if isinstance(match, str) else match[0]

                    # Map value type to claim type
                    claim_type_map = {
                        "learning_rate": ClaimType.HYPERPARAMETER,
                        "batch_size": ClaimType.HYPERPARAMETER,
                        "epochs": ClaimType.TRAINING_PROCEDURE,
                        "cv_folds": ClaimType.DATA_SPLIT,
                        "dropout": ClaimType.MODEL_ARCHITECTURE,
                        "weight_decay": ClaimType.HYPERPARAMETER,
                    }

                    claim_type = claim_type_map.get(value_type, ClaimType.HYPERPARAMETER)

                    # Generate human-readable claim text
                    claim_text_map = {
                        "learning_rate": f"Learning rate is {value}",
                        "batch_size": f"Batch size is {value}",
                        "epochs": f"Training for {value} epochs",
                        "cv_folds": f"Uses {value}-fold cross-validation",
                        "dropout": f"Dropout rate is {value}",
                        "weight_decay": f"Weight decay is {value}",
                    }

                    claim = Claim(
                        id=str(uuid.uuid4()),
                        text=claim_text_map.get(value_type, f"{value_type}: {value}"),
                        claim_type=claim_type,
                        keywords=[value_type.replace("_", " "), value],
                        expected_value=value,
                        source_location="pattern-extracted",
                        verifiable=True,
                    )
                    claims.append(claim)
                    break  # Only take first match per pattern type

        return claims

    async def _single_pass_extraction(self, text: str) -> list[Claim]:
        """Extract claims in a single LLM pass."""
        # Truncate if needed
        max_chars = 15000
        if len(text) > max_chars:
            text = self._extract_relevant_sections(text, max_chars)

        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)

        response = await self.ai_provider.complete(
            prompt=prompt,
        )

        return self._parse_claims(response)

    def _is_important_section(self, section_name: str) -> bool:
        """Check if a section is important for claim extraction."""
        name_lower = section_name.lower()

        # Skip if matches skip keywords
        for skip_keyword in SKIP_SECTION_KEYWORDS:
            if skip_keyword in name_lower:
                return False

        # Keep if matches important keywords
        for important_keyword in IMPORTANT_SECTION_KEYWORDS:
            if important_keyword in name_lower:
                return True

        # For numbered sections like "3 Methods" or "4.1 Training Details"
        # Strip the number prefix and check again
        stripped = re.sub(r"^\d+\.?\d*\s*", "", name_lower).strip()
        for important_keyword in IMPORTANT_SECTION_KEYWORDS:
            if important_keyword in stripped:
                return True

        # If section name is generic (like "Section 1"), keep it
        # but only if it doesn't match skip keywords
        if section_name.startswith("Section "):
            return True

        # Default: skip unknown sections to be conservative
        return False

    async def _multi_pass_extraction(self, text: str, max_concurrent: int = 5) -> list[Claim]:
        """Extract claims in multiple passes for long documents (parallelized)."""
        # Identify sections
        sections = self._split_into_sections(text)
        logger.debug(f"Split document into {len(sections)} sections")

        # Filter to important sections only
        important_sections = [
            (name, section_text) for name, section_text in sections
            if self._is_important_section(name) and len(section_text.strip()) >= 100
        ]
        logger.info(
            f"Filtered to {len(important_sections)} important sections "
            f"(from {len(sections)} total)"
        )

        # Process sections in parallel with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_section(section_name: str, section_text: str):
            async with semaphore:
                # Truncate each section
                section_text = section_text[:8000]
                prompt = CLAIM_EXTRACTION_PROMPT.format(text=section_text)

                try:
                    response = await self.ai_provider.complete(prompt=prompt)
                    section_claims = self._parse_claims(response)

                    # Add source location
                    for claim in section_claims:
                        if not claim.source_location:
                            claim.source_location = section_name

                    return section_claims
                except Exception as e:
                    logger.warning(f"Failed to extract claims from section {section_name}: {e}")
                    return []

        tasks = [
            extract_section(name, section_text)
            for name, section_text in important_sections
        ]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_claims = []
        for section_claims in results:
            all_claims.extend(section_claims)

        return all_claims

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split document into logical sections."""
        sections = []

        # Common section headers
        section_patterns = [
            r"(?:^|\n)#+\s*(.+?)(?:\n|$)",  # Markdown headers
            r"(?:^|\n)(\d+\.?\s*[A-Z][^.\n]+)(?:\n|$)",  # Numbered sections
            r"(?:^|\n)([A-Z][A-Z\s]+)(?:\n|$)",  # ALL CAPS headers
        ]

        # First, try to split by headers
        current_section = "Introduction"
        current_text = []

        for line in text.split("\n"):
            # Check if this is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, "\n" + line)
                if match:
                    # Save previous section
                    if current_text:
                        sections.append((current_section, "\n".join(current_text)))
                    current_section = match.group(1).strip()
                    current_text = []
                    is_header = True
                    break

            if not is_header:
                current_text.append(line)

        # Add final section
        if current_text:
            sections.append((current_section, "\n".join(current_text)))

        # If no sections found, return whole text
        if len(sections) <= 1:
            # Fall back to chunking by character count
            chunk_size = 6000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                sections.append((f"Section {i // chunk_size + 1}", chunk))

        return sections

    def _deduplicate_claims(self, claims: list[Claim]) -> list[Claim]:
        """Remove duplicate claims, keeping the most specific version."""
        unique_claims = {}

        for claim in claims:
            # Create a key based on claim type and expected value
            if claim.expected_value:
                key = (claim.claim_type, claim.expected_value)
            else:
                # For claims without expected values, use normalized text
                key = (claim.claim_type, self._normalize_claim_text(claim.text))

            if key not in unique_claims:
                unique_claims[key] = claim
            else:
                # Keep the claim with more keywords or longer text
                existing = unique_claims[key]
                if len(claim.keywords) > len(existing.keywords) or len(claim.text) > len(existing.text):
                    unique_claims[key] = claim

        return list(unique_claims.values())

    def _normalize_claim_text(self, text: str) -> str:
        """Normalize claim text for deduplication."""
        # Lowercase, remove extra whitespace, remove punctuation
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text[:100]  # Truncate for comparison

    def _parse_claims(self, response: str) -> list[Claim]:
        """Parse LLM response into Claim objects."""
        claims = []

        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            if not isinstance(data, list):
                data = [data]

            for item in data:
                claim_type_str = item.get("claim_type", "other")
                try:
                    claim_type = ClaimType(claim_type_str)
                except ValueError:
                    claim_type = ClaimType.OTHER

                # Convert expected_value to string if it's not None
                expected_value = item.get("expected_value")
                if expected_value is not None:
                    expected_value = str(expected_value)

                # Ensure keywords are strings
                keywords = item.get("keywords", [])
                if isinstance(keywords, list):
                    keywords = [str(k) for k in keywords]
                else:
                    keywords = []

                # Ensure source_location is a string
                source_location = item.get("source_location")
                if source_location is not None:
                    source_location = str(source_location)

                claim = Claim(
                    id=str(uuid.uuid4()),
                    text=str(item.get("text", "")),
                    claim_type=claim_type,
                    keywords=keywords,
                    expected_value=expected_value,
                    source_location=source_location,
                    verifiable=True,
                )
                claims.append(claim)

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract claims manually
            # This is a fallback for malformed responses
            pass

        return claims

    def _extract_relevant_sections(self, text: str, max_chars: int) -> str:
        """Extract methodology and experiments sections from long text."""
        # Common section headers to look for
        section_markers = [
            "methodology", "methods", "method", "approach",
            "experiments", "experimental", "implementation",
            "training", "evaluation", "setup", "configuration",
            "architecture", "model", "data", "dataset",
        ]

        lines = text.split("\n")
        relevant_lines = []
        in_relevant_section = False
        chars_collected = 0

        # Always include the first part (abstract/intro often has key claims)
        intro_chars = min(3000, max_chars // 3)
        intro = text[:intro_chars]
        relevant_lines.append(intro)
        chars_collected += len(intro)

        # Look for relevant sections
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if this is a section header
            is_header = any(marker in line_lower for marker in section_markers)

            if is_header:
                in_relevant_section = True

            if in_relevant_section:
                if chars_collected + len(line) > max_chars:
                    break
                relevant_lines.append(line)
                chars_collected += len(line) + 1

                # Stop if we hit references or appendix
                if any(x in line_lower for x in ["references", "appendix", "acknowledgment"]):
                    break

        return "\n".join(relevant_lines)
