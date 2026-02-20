"""Analyze code behaviors to understand what the code actually does."""

import asyncio
import json
import re
import uuid
from typing import Optional

from backend.models.audit import CodeBehavior, ClaimType
from backend.models.codebase import Codebase
from backend.services.ai import get_ai_provider
from backend.services.semantic_chunking import SemanticChunker
from backend.services.logging import logger


# Pattern-based extraction for common ML values (reduces LLM calls)
ML_PATTERNS = {
    # Learning rate patterns
    "learning_rate": [
        r"(?:learning_rate|lr)\s*[=:]\s*([0-9.e\-]+)",
        r"lr\s*=\s*([0-9.e\-]+)",
        r"['\"]lr['\"]\s*:\s*([0-9.e\-]+)",
        r"['\"]learning_rate['\"]\s*:\s*([0-9.e\-]+)",
    ],
    # Batch size patterns
    "batch_size": [
        r"batch_size\s*[=:]\s*(\d+)",
        r"['\"]batch_size['\"]\s*:\s*(\d+)",
    ],
    # Epochs patterns
    "epochs": [
        r"(?:num_)?epochs?\s*[=:]\s*(\d+)",
        r"['\"](?:num_)?epochs?['\"]\s*:\s*(\d+)",
    ],
    # Cross-validation folds
    "cv_folds": [
        r"(?:n_splits|n_folds|cv|k_fold)\s*[=:]\s*(\d+)",
        r"KFold\s*\(\s*(?:n_splits\s*=\s*)?(\d+)",
        r"StratifiedKFold\s*\(\s*(?:n_splits\s*=\s*)?(\d+)",
        r"cross_val_score\s*\([^)]*cv\s*=\s*(\d+)",
    ],
    # Train/test split
    "test_split": [
        r"test_size\s*[=:]\s*([0-9.]+)",
        r"train_test_split\s*\([^)]*test_size\s*=\s*([0-9.]+)",
    ],
    # Optimizer
    "optimizer": [
        r"(?:optim\.)?(\w+)\s*\(\s*(?:model\.)?parameters\(\)",
        r"optimizer\s*[=:]\s*['\"]?(\w+)['\"]?",
    ],
    # Dropout
    "dropout": [
        r"(?:nn\.)?Dropout\s*\(\s*(?:p\s*=\s*)?([0-9.]+)",
        r"dropout(?:_rate|_prob)?\s*[=:]\s*([0-9.]+)",
    ],
    # Hidden dimensions / units
    "hidden_size": [
        r"hidden_(?:size|dim|units?)\s*[=:]\s*(\d+)",
        r"['\"]hidden_(?:size|dim|units?)['\"]\s*:\s*(\d+)",
    ],
    # Weight decay / L2 regularization
    "weight_decay": [
        r"weight_decay\s*[=:]\s*([0-9.e\-]+)",
        r"l2\s*[=:]\s*([0-9.e\-]+)",
    ],
    # Random seed
    "random_seed": [
        r"(?:random_)?seed\s*[=:]\s*(\d+)",
        r"set_seed\s*\(\s*(\d+)\s*\)",
        r"manual_seed\s*\(\s*(\d+)\s*\)",
        r"np\.random\.seed\s*\(\s*(\d+)\s*\)",
    ],
}

# Behavior type detection patterns
BEHAVIOR_PATTERNS = {
    ClaimType.DATA_SPLIT: [
        r"train_test_split",
        r"KFold|StratifiedKFold|cross_val",
        r"\.split\(",
    ],
    ClaimType.MODEL_ARCHITECTURE: [
        r"class\s+\w+\s*\(\s*(?:nn\.Module|Model|keras\.Model)",
        r"Sequential\s*\(",
        r"def\s+(?:forward|call|build)\s*\(",
    ],
    ClaimType.TRAINING_PROCEDURE: [
        r"\.train\s*\(",
        r"optimizer\.step\s*\(",
        r"loss\.backward\s*\(",
        r"\.fit\s*\(",
        r"for\s+(?:epoch|e)\s+in",
    ],
    ClaimType.EVALUATION_METRICS: [
        r"(?:accuracy|precision|recall|f1)_score",
        r"roc_auc|auc_roc",
        r"confusion_matrix",
        r"classification_report",
        r"\.eval\s*\(",
    ],
    ClaimType.DATA_PREPROCESSING: [
        r"\.transform\s*\(",
        r"normalize|standardize|scale",
        r"fillna|dropna|impute",
        r"Tokenizer|preprocess",
    ],
}


CODE_ANALYSIS_PROMPT = """Analyze this code and describe what it actually does. Focus on specific, verifiable details.

Code from {file_path}:
```{language}
{code}
```

Determine:
1. behavior_type: What category does this code fall into?
   - data_preprocessing: Data loading, cleaning, transformation
   - data_split: Train/val/test splitting, cross-validation setup
   - model_architecture: Model definition, layers, structure
   - training_procedure: Training loop, optimizer setup, learning rate
   - evaluation_metrics: Metric computation, evaluation logic
   - hyperparameter: Hyperparameter definitions
   - implementation_detail: Other implementation specifics

2. description: A clear description of what this code does. Be specific about:
   - Exact values (learning rate, batch size, epochs, etc.)
   - Specific functions/methods used
   - Data transformations applied
   - Split ratios if applicable

3. actual_value: If this code sets a specific value (like learning rate, fold count, etc.),
   extract that value as a string.

Return JSON with: behavior_type, description, actual_value (or null if not applicable)

Return ONLY valid JSON, no other text."""


class CodeAnalyzer:
    """Analyze codebase to extract actual behaviors."""

    def __init__(self, use_pattern_detection: bool = True):
        self.ai_provider = None
        self.chunker = SemanticChunker()
        self.use_pattern_detection = use_pattern_detection
        # Compile patterns for efficiency
        self._compiled_ml_patterns = {
            key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in ML_PATTERNS.items()
        }
        self._compiled_behavior_patterns = {
            key: [re.compile(p) for p in patterns]
            for key, patterns in BEHAVIOR_PATTERNS.items()
        }

    def _extract_pattern_values(self, code: str) -> dict[str, str]:
        """Extract values from code using pattern matching."""
        extracted = {}
        for key, patterns in self._compiled_ml_patterns.items():
            for pattern in patterns:
                match = pattern.search(code)
                if match:
                    extracted[key] = match.group(1)
                    break
        return extracted

    def _detect_behavior_type(self, code: str) -> Optional[ClaimType]:
        """Detect behavior type from code patterns."""
        type_scores = {}
        for claim_type, patterns in self._compiled_behavior_patterns.items():
            score = sum(1 for p in patterns if p.search(code))
            if score > 0:
                type_scores[claim_type] = score

        if type_scores:
            return max(type_scores, key=type_scores.get)
        return None

    def _build_pattern_behavior(
        self,
        code: str,
        file_path: str,
        relative_path: str,
        start_line: int,
        end_line: int,
        chunk_id: str,
    ) -> Optional[CodeBehavior]:
        """Build a CodeBehavior from pattern detection only (no LLM)."""
        extracted = self._extract_pattern_values(code)
        behavior_type = self._detect_behavior_type(code)

        if not extracted and not behavior_type:
            return None

        # Build description from extracted values
        desc_parts = []
        if behavior_type:
            desc_parts.append(f"Type: {behavior_type.value}")

        for key, value in extracted.items():
            readable_key = key.replace("_", " ")
            desc_parts.append(f"{readable_key}: {value}")

        # Pick the most significant extracted value
        actual_value = None
        priority_keys = ["cv_folds", "learning_rate", "batch_size", "epochs", "test_split"]
        for key in priority_keys:
            if key in extracted:
                actual_value = extracted[key]
                break

        if not actual_value and extracted:
            actual_value = list(extracted.values())[0]

        return CodeBehavior(
            chunk_id=chunk_id,
            file_path=file_path,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            behavior_type=behavior_type or ClaimType.IMPLEMENTATION_DETAIL,
            description="; ".join(desc_parts) if desc_parts else "Pattern-detected behavior",
            code_snippet=code[:1000],
            actual_value=actual_value,
            extracted_values=extracted,  # Store all extracted values
        )

    async def analyze_codebase(
        self,
        codebase: Codebase,
        claim_types: Optional[list[ClaimType]] = None,
        max_concurrent: int = 5,
    ) -> list[CodeBehavior]:
        """
        Analyze codebase to extract behaviors.

        Args:
            codebase: The codebase to analyze
            claim_types: Types of claims to focus on (for targeted analysis)
            max_concurrent: Maximum concurrent LLM calls

        Returns:
            List of CodeBehavior objects
        """
        if self.ai_provider is None:
            self.ai_provider = get_ai_provider()

        # Identify relevant files based on claim types
        relevant_files = self._identify_relevant_files(codebase, claim_types)

        # Collect all chunks from all files
        all_chunks = []
        for file in relevant_files:
            chunks = self.chunker.chunk_file(file)
            for chunk in chunks:
                all_chunks.append((file, chunk))

        # Analyze all chunks in parallel with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(file, chunk):
            async with semaphore:
                return await self._analyze_chunk(
                    code=chunk.content,
                    file_path=file.path,
                    relative_path=file.relative_path,
                    language=file.language,
                    start_line=chunk.metadata.start_line,
                    end_line=chunk.metadata.end_line,
                    chunk_id=chunk.chunk_id,
                )

        tasks = [analyze_with_limit(file, chunk) for file, chunk in all_chunks]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        behaviors = [b for b in results if b is not None]

        return behaviors

    async def _analyze_chunk(
        self,
        code: str,
        file_path: str,
        relative_path: str,
        language: str,
        start_line: int,
        end_line: int,
        chunk_id: str,
        use_llm_fallback: bool = True,
    ) -> Optional[CodeBehavior]:
        """Analyze a single code chunk.

        First tries pattern-based detection for common ML patterns.
        Falls back to LLM for more detailed analysis if patterns don't match.
        """
        # Skip very short chunks
        if len(code.strip()) < 50:
            return None

        # Skip test files for now (could be configurable)
        if "test" in relative_path.lower() and "test" not in file_path.split("/")[-1].lower():
            return None

        # Try pattern-based detection first (faster, no API calls)
        if self.use_pattern_detection:
            pattern_behavior = self._build_pattern_behavior(
                code=code,
                file_path=file_path,
                relative_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                chunk_id=chunk_id,
            )
            if pattern_behavior and pattern_behavior.extracted_values:
                # Pattern detection found meaningful values
                pattern_behavior.from_pattern = True
                logger.debug(f"Pattern detected values in {relative_path}: {pattern_behavior.extracted_values}")
                return pattern_behavior

        # Fall back to LLM for detailed analysis
        if not use_llm_fallback:
            return None

        prompt = CODE_ANALYSIS_PROMPT.format(
            file_path=relative_path,
            language=language,
            code=code[:3000],  # Truncate very long code
        )

        try:
            response = await self.ai_provider.complete(
                prompt=prompt, temperature=0.0,
            )

            data = self._parse_response(response)
            if not data:
                return None

            behavior_type_str = data.get("behavior_type", "implementation_detail")
            try:
                behavior_type = ClaimType(behavior_type_str)
            except ValueError:
                behavior_type = ClaimType.IMPLEMENTATION_DETAIL

            return CodeBehavior(
                chunk_id=chunk_id,
                file_path=file_path,
                relative_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                behavior_type=behavior_type,
                description=data.get("description", ""),
                code_snippet=code[:1000],  # Store truncated snippet
                actual_value=data.get("actual_value"),
                from_pattern=False,
            )

        except Exception as e:
            logger.debug(f"LLM analysis failed for {relative_path}: {e}")
            return None

    def _parse_response(self, response: str) -> Optional[dict]:
        """Parse LLM response into dict."""
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _identify_relevant_files(
        self,
        codebase: Codebase,
        claim_types: Optional[list[ClaimType]],
    ) -> list:
        """Identify files likely to contain relevant code."""
        # Keywords that suggest a file is relevant for different claim types
        relevance_keywords = {
            ClaimType.DATA_PREPROCESSING: ["data", "preprocess", "transform", "load", "dataset", "augment"],
            ClaimType.DATA_SPLIT: ["split", "train", "val", "test", "fold", "cv", "cross"],
            ClaimType.MODEL_ARCHITECTURE: ["model", "network", "net", "arch", "backbone", "encoder", "decoder"],
            ClaimType.TRAINING_PROCEDURE: ["train", "fit", "optim", "learn", "loss", "epoch"],
            ClaimType.EVALUATION_METRICS: ["eval", "metric", "score", "accuracy", "loss", "test", "valid"],
            ClaimType.HYPERPARAMETER: ["config", "param", "hparam", "setting", "args"],
        }

        if claim_types is None:
            # Return all Python files if no filter
            return [f for f in codebase.files if f.language == "python"]

        # Collect relevant keywords
        keywords = set()
        for ct in claim_types:
            if ct in relevance_keywords:
                keywords.update(relevance_keywords[ct])

        # Filter files
        relevant = []
        for file in codebase.files:
            if file.language != "python":
                continue

            file_lower = file.relative_path.lower()

            # Check if filename matches any keywords
            if any(kw in file_lower for kw in keywords):
                relevant.append(file)
            # Also check file content for keywords
            elif any(kw in file.content.lower()[:5000] for kw in keywords):
                relevant.append(file)

        # If no matches, return main files
        if not relevant:
            main_files = ["main", "train", "run", "experiment"]
            for file in codebase.files:
                if any(m in file.relative_path.lower() for m in main_files):
                    relevant.append(file)

        return relevant[:10]  # Limit to avoid too many API calls
