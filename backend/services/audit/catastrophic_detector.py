"""Detect catastrophic patterns that invalidate results."""

import asyncio
import json
import re
import uuid
from typing import Optional

from backend.models.audit import (
    CatastrophicCategory,
    CatastrophicWarning,
    IssueTier,
)
from backend.models.codebase import Codebase, CodeFile
from backend.services.ai import get_ai_provider
from backend.services.logging import logger


# Pattern-based detection rules

# Category 1: Data Leakage Patterns
LEAKAGE_PATTERNS = [
    {
        "name": "fit_before_split",
        "pattern": r"\.fit\s*\([^)]*\)[\s\S]{0,1000}train_test_split",
        "description": "Fitting (normalization/scaling) happens before train/test split",
        "why": "Statistics computed on full data leak information about test set into training",
        "recommendation": "Move fit() call after the split, fitting only on training data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "fit_transform_before_split",
        "pattern": r"\.fit_transform\s*\([^)]*\)[\s\S]{0,1000}train_test_split",
        "description": "fit_transform() called before train/test split",
        "why": "Transformation learned on full data, leaking test information",
        "recommendation": "Split first, then fit_transform on train and transform on test",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "scaler_before_split",
        "pattern": r"(StandardScaler|MinMaxScaler|RobustScaler|Normalizer)\s*\(\s*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "Scaler fitted before data split",
        "why": "Test set statistics influence the scaling, causing data leakage",
        "recommendation": "Create scaler, split data, then fit on training data only",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "feature_selection_before_split",
        "pattern": r"(SelectKBest|SelectFromModel|RFE|VarianceThreshold)\s*\([^)]*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "Feature selection performed before train/test split",
        "why": "Feature importance computed using test data, leaking information",
        "recommendation": "Split first, then perform feature selection on training data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "pca_before_split",
        "pattern": r"(PCA|TruncatedSVD|UMAP|TSNE)\s*\([^)]*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "Dimensionality reduction fitted before train/test split",
        "why": "Test data contributes to principal components, causing leakage",
        "recommendation": "Fit PCA/UMAP only on training data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "imputer_before_split",
        "pattern": r"(SimpleImputer|KNNImputer|IterativeImputer)\s*\([^)]*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "Imputer fitted before train/test split",
        "why": "Missing value statistics computed from test data",
        "recommendation": "Fit imputer only on training data after splitting",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "cv_preprocessing_outside",
        "pattern": r"(\.fit_transform|\.fit)\s*\([^)]*\)[\s\S]{0,500}(cross_val_score|KFold|StratifiedKFold)",
        "description": "Preprocessing performed before cross-validation loop",
        "why": "Each CV fold trains on data that includes validation fold statistics",
        "recommendation": "Use Pipeline to ensure preprocessing happens inside CV folds",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "target_encoding_leakage",
        "pattern": r"(groupby|merge|join)[\s\S]{0,200}(mean|median|std)\s*\([\s\S]{0,500}(y|target|label)",
        "description": "Target-based feature encoding may cause leakage",
        "why": "Target statistics from test data may be used in features",
        "recommendation": "Compute target statistics only from training data",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
]

# Category 2: Evaluation Error Patterns
EVALUATION_ERROR_PATTERNS = [
    {
        "name": "eval_on_train_data",
        "pattern": r"(accuracy_score|f1_score|precision_score|recall_score|roc_auc_score)\s*\(\s*(y_train|train_y)",
        "description": "Metric computed on training data labels",
        "why": "Evaluating on training data gives misleadingly optimistic results",
        "recommendation": "Compute metrics on test/validation data only",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "predict_on_train",
        "pattern": r"\.predict\s*\(\s*(X_train|train_X)[\s\S]{0,200}(score|accuracy|metric)",
        "description": "Predictions on training data used for evaluation",
        "why": "Training accuracy is not a reliable measure of model performance",
        "recommendation": "Evaluate using held-out test or validation data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "test_in_gridsearch",
        "pattern": r"(GridSearchCV|RandomizedSearchCV)\s*\([^)]*\)[\s\S]{0,500}\.fit\s*\([^,]*,\s*[^,]*\)[\s\S]{0,200}(X_test|test_X)",
        "description": "Test set may be used in hyperparameter search",
        "why": "Using test set for tuning leads to overfitting on test data",
        "recommendation": "Use only training data with CV for hyperparameter tuning",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "accuracy_imbalanced",
        "pattern": r"accuracy_score[\s\S]{0,500}(imbalance|class_weight|oversamp|undersamp|SMOTE)",
        "description": "Using accuracy metric on potentially imbalanced data",
        "why": "Accuracy is misleading for imbalanced datasets",
        "recommendation": "Use F1, precision-recall, or balanced accuracy instead",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "no_stratification",
        "pattern": r"train_test_split\s*\([^)]*(?!stratify)[^)]*\)",
        "description": "train_test_split without stratification",
        "why": "Class distribution may differ between train and test sets",
        "recommendation": "Use stratify=y parameter for classification tasks",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
]

# Category 3: Training Issue Patterns
TRAINING_ISSUE_PATTERNS = [
    {
        "name": "no_model_eval_mode",
        "pattern": r"(valid|test|eval)[\s\S]{0,300}model\s*\([\s\S]{0,100}(?!\.eval\(\))",
        "description": "Model may not be in eval mode during validation/testing",
        "why": "BatchNorm and Dropout behave differently in train vs eval mode",
        "recommendation": "Call model.eval() before validation/testing",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
        "requires_context": True,
    },
    {
        "name": "no_zero_grad",
        "pattern": r"for\s+[\w,\s]+\s+in[\s\S]{0,200}loss\.backward\s*\(\)[\s\S]{0,100}optimizer\.step\s*\(\)[\s\S]{0,100}(?!zero_grad)",
        "description": "Training loop may not zero gradients",
        "why": "Gradients accumulate if not zeroed, causing incorrect updates",
        "recommendation": "Call optimizer.zero_grad() before loss.backward()",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "backward_after_step",
        "pattern": r"optimizer\.step\s*\(\)[\s\S]{0,50}loss\.backward\s*\(\)",
        "description": "optimizer.step() called before loss.backward()",
        "why": "Gradients must be computed before optimizer step",
        "recommendation": "Call loss.backward() before optimizer.step()",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "mse_for_classification",
        "pattern": r"(MSELoss|mse_loss|mean_squared_error)[\s\S]{0,500}(CrossEntropy|softmax|sigmoid|classification|class_)",
        "description": "MSE loss may be used for classification task",
        "why": "MSE is inappropriate for classification, use cross-entropy",
        "recommendation": "Use CrossEntropyLoss or BCELoss for classification",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "labels_not_converted",
        "pattern": r"(CrossEntropyLoss|NLLLoss)[\s\S]{0,500}(\.float\(\)|\.astype\(float\))",
        "description": "Labels converted to float for cross-entropy loss",
        "why": "CrossEntropyLoss expects integer class indices, not floats",
        "recommendation": "Keep labels as integers (Long tensor in PyTorch)",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "test_augmentation",
        "pattern": r"(test|valid|eval)[\s\S]{0,200}(RandomHorizontalFlip|RandomRotation|RandomCrop|augment)",
        "description": "Data augmentation may be applied to test/validation data",
        "why": "Augmentation should only be applied during training",
        "recommendation": "Disable augmentation for validation and test sets",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "no_shuffle_training",
        "pattern": r"DataLoader\s*\([^)]*train[\s\S]{0,100}shuffle\s*=\s*False",
        "description": "Training DataLoader with shuffle=False",
        "why": "Not shuffling training data can cause learning issues",
        "recommendation": "Set shuffle=True for training DataLoader",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
]

# Category 4-7 patterns (to be checked in Phase 4)
REPRODUCIBILITY_PATTERNS = [
    {
        "name": "no_random_seed",
        "pattern": r"(np\.random|random\.|torch\.|tf\.random)",
        "description": "Random operations detected but no seed setting found",
        "why": "Results may not be reproducible",
        "recommendation": "Set random seeds: random.seed(), np.random.seed(), torch.manual_seed()",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
        "requires_absence": ["random.seed", "np.random.seed", "torch.manual_seed", "set_seed", "SEED"],
    },
    {
        "name": "non_deterministic_cudnn",
        "pattern": r"torch\.cuda[\s\S]{0,500}(?!torch\.backends\.cudnn\.deterministic\s*=\s*True)",
        "description": "CUDA used without deterministic settings",
        "why": "CUDA operations can be non-deterministic by default",
        "recommendation": "Set torch.backends.cudnn.deterministic = True",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
    {
        "name": "hardcoded_path",
        "pattern": r"['\"][/\\](?:home|Users|var|tmp|data)[/\\][^'\"]+['\"]",
        "description": "Hardcoded absolute path detected",
        "why": "Absolute paths break reproducibility on other machines",
        "recommendation": "Use relative paths or environment variables",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
]

DATA_INTEGRITY_PATTERNS = [
    {
        "name": "nan_not_handled",
        "pattern": r"(pd\.read|np\.load|torch\.load)[\s\S]{0,1000}(?!dropna|fillna|isna|nan_to_num|isnan)",
        "description": "Data loaded but NaN handling not detected nearby",
        "why": "NaN values can propagate and corrupt model training",
        "recommendation": "Check for and handle NaN values after loading data",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
        "requires_context": True,
    },
    {
        "name": "int_truncation",
        "pattern": r"\.astype\s*\(\s*(int|np\.int)",
        "description": "Float to int conversion may cause data loss",
        "why": "Truncating floats to integers loses precision",
        "recommendation": "Use round() before astype(int) if intentional",
        "tier": IssueTier.MINOR_DISCREPANCY,
    },
]

STATISTICAL_ERROR_PATTERNS = [
    {
        "name": "single_run",
        "pattern": r"(accuracy|f1|score|loss)\s*=[\s\S]{0,1000}(?!for\s+|mean|std|average|repeat)",
        "description": "Results may be from single run without aggregation",
        "why": "Single run results have high variance, not reliable",
        "recommendation": "Run multiple times and report mean +/- std",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
        "requires_context": True,
    },
    {
        "name": "no_confidence_interval",
        "pattern": r"print\s*\([^)]*(?:accuracy|f1|score)[\s\S]{0,200}(?!std|±|\\+/-|confidence)",
        "description": "Results printed without confidence intervals",
        "why": "Point estimates without uncertainty can be misleading",
        "recommendation": "Report standard deviation or confidence intervals",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
]

# Category 7: Time Series Specific Patterns
TIME_SERIES_PATTERNS = [
    {
        "name": "kfold_timeseries",
        "pattern": r"(KFold|StratifiedKFold)\s*\([^)]*\)[\s\S]{0,500}(time|date|datetime|timestamp)",
        "description": "Using KFold instead of TimeSeriesSplit for temporal data",
        "why": "KFold shuffles data, causing future data to leak into training",
        "recommendation": "Use TimeSeriesSplit for time-ordered data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "shuffle_timeseries",
        "pattern": r"(shuffle\s*=\s*True|\.shuffle\s*\(\))[\s\S]{0,500}(time|date|datetime|timestamp|sequence)",
        "description": "Shuffling potentially time-ordered data",
        "why": "Shuffling time series breaks temporal ordering and causes leakage",
        "recommendation": "Disable shuffling for time series data",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "future_features",
        "pattern": r"(shift\s*\(\s*-|\.shift\s*\(\s*-\d+\)|lead|forward_fill)[\s\S]{0,200}(feature|X_|train)",
        "description": "Possible use of future data as features",
        "why": "Using future values as features causes temporal leakage",
        "recommendation": "Only use past/current values as features (positive shift)",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "random_split_timeseries",
        "pattern": r"train_test_split[\s\S]{0,500}(time|date|datetime|timestamp)",
        "description": "Random train/test split on time series data",
        "why": "Random splitting breaks temporal order, leaking future info",
        "recommendation": "Split by time point: all training data before test data",
        "tier": IssueTier.RESULTS_INVALID,
    },
]

# Category 8: Deep Learning Specific Patterns
DEEP_LEARNING_PATTERNS = [
    {
        "name": "no_gradient_clipping_rnn",
        "pattern": r"(LSTM|GRU|RNN)\s*\([^)]*\)[\s\S]{0,1000}(?!clip_grad|grad_clip|max_norm)",
        "description": "RNN without gradient clipping detected",
        "why": "RNNs are prone to exploding gradients without clipping",
        "recommendation": "Add torch.nn.utils.clip_grad_norm_() or clip_grad_value_()",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "batchnorm_small_batch",
        "pattern": r"BatchNorm(1d|2d|3d)?\s*\([^)]*\)[\s\S]{0,500}batch_size\s*=\s*([1-8])\b",
        "description": "BatchNorm with very small batch size",
        "why": "BatchNorm statistics are unreliable with batch_size < 8",
        "recommendation": "Use GroupNorm or LayerNorm for small batches, or increase batch size",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "no_detach_hidden",
        "pattern": r"(hidden|h_0|c_0|cell)\s*=[\s\S]{0,200}(LSTM|GRU|RNN)[\s\S]{0,500}(?!\.detach\(\)|detach\()",
        "description": "RNN hidden states may not be detached between sequences",
        "why": "Not detaching causes gradients to flow across sequences, memory issues",
        "recommendation": "Detach hidden states: hidden = hidden.detach()",
        "tier": IssueTier.REPRODUCIBILITY_RISK,
    },
    {
        "name": "frozen_pretrained_wrong",
        "pattern": r"(requires_grad\s*=\s*False|\.freeze\(\))[\s\S]{0,200}(optimizer|\.backward\(\))",
        "description": "Frozen layers included in optimizer or backward pass",
        "why": "Frozen parameters shouldn't be optimized, wastes computation",
        "recommendation": "Exclude frozen params from optimizer: filter(lambda p: p.requires_grad, model.parameters())",
        "tier": IssueTier.MINOR_DISCREPANCY,
    },
    {
        "name": "softmax_with_crossentropy",
        "pattern": r"(Softmax|softmax)\s*\([^)]*\)[\s\S]{0,300}(CrossEntropyLoss|cross_entropy)",
        "description": "Applying softmax before CrossEntropyLoss",
        "why": "CrossEntropyLoss includes softmax internally, double softmax is wrong",
        "recommendation": "Remove softmax layer, CrossEntropyLoss expects raw logits",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "sigmoid_with_bcewithlogits",
        "pattern": r"(Sigmoid|sigmoid)\s*\([^)]*\)[\s\S]{0,300}(BCEWithLogitsLoss|binary_cross_entropy_with_logits)",
        "description": "Applying sigmoid before BCEWithLogitsLoss",
        "why": "BCEWithLogitsLoss includes sigmoid, double sigmoid is wrong",
        "recommendation": "Remove sigmoid layer, or use BCELoss instead",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "no_torch_no_grad",
        "pattern": r"def\s+(eval|test|valid|predict|inference)\s*\([^)]*\)[\s\S]{0,500}model\s*\([\s\S]{0,500}(?!torch\.no_grad|@torch\.no_grad|with\s+no_grad)",
        "description": "Inference without torch.no_grad() context",
        "why": "Unnecessary gradient computation wastes memory and time",
        "recommendation": "Wrap inference in: with torch.no_grad():",
        "tier": IssueTier.MINOR_DISCREPANCY,
    },
]

# Category 9: Scikit-learn Specific Patterns
SKLEARN_PATTERNS = [
    {
        "name": "label_encoder_before_split",
        "pattern": r"LabelEncoder\s*\(\s*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "LabelEncoder fitted before train/test split",
        "why": "Test labels may influence encoding if new classes appear",
        "recommendation": "Fit LabelEncoder on training data only",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "ordinal_encoder_before_split",
        "pattern": r"OrdinalEncoder\s*\(\s*\)[\s\S]{0,500}\.fit[\s\S]{0,500}train_test_split",
        "description": "OrdinalEncoder fitted before train/test split",
        "why": "Categories from test set may influence encoding",
        "recommendation": "Fit OrdinalEncoder on training data only",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "smote_before_split",
        "pattern": r"(SMOTE|ADASYN|BorderlineSMOTE)\s*\([^)]*\)[\s\S]{0,500}train_test_split",
        "description": "Oversampling (SMOTE) before train/test split",
        "why": "Synthetic samples may contain test set information",
        "recommendation": "Apply SMOTE only to training data after splitting",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "test_data_in_pipeline_fit",
        "pattern": r"Pipeline\s*\([^)]*\)[\s\S]{0,500}\.fit\s*\(\s*(X\s*,|data\s*,)[\s\S]{0,200}train_test_split",
        "description": "Pipeline fitted on full data before split",
        "why": "All preprocessing steps in pipeline see test data",
        "recommendation": "Split first, then fit pipeline on training data only",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "different_preprocessing_inference",
        "pattern": r"\.transform\s*\([\s\S]{0,500}\.fit_transform\s*\(",
        "description": "Possible inconsistent preprocessing between train and inference",
        "why": "Different preprocessing leads to train-test skew",
        "recommendation": "Use same fitted transformer for both: fit on train, transform on test",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
        "requires_context": True,
    },
]

# Category 10: Metric and Evaluation Patterns
METRIC_PATTERNS = [
    {
        "name": "classification_metric_regression",
        "pattern": r"(accuracy_score|f1_score|precision_score|recall_score)[\s\S]{0,500}(regress|continuous|MSE|MAE|RMSE)",
        "description": "Classification metrics used for regression task",
        "why": "Classification metrics are meaningless for continuous outputs",
        "recommendation": "Use regression metrics: MSE, MAE, R², RMSE",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "regression_metric_classification",
        "pattern": r"(mean_squared_error|mean_absolute_error|r2_score)[\s\S]{0,500}(classif|categorical|softmax|sigmoid|CrossEntropy)",
        "description": "Regression metrics used for classification task",
        "why": "Regression metrics don't properly evaluate classification",
        "recommendation": "Use classification metrics: accuracy, F1, AUC-ROC",
        "tier": IssueTier.RESULTS_INVALID,
    },
    {
        "name": "auc_multiclass_wrong",
        "pattern": r"roc_auc_score\s*\([^)]*(?!multi_class|average)[^)]*\)[\s\S]{0,200}(num_classes|n_classes)\s*[>=]\s*3",
        "description": "AUC-ROC for multiclass without proper averaging",
        "why": "AUC-ROC needs multi_class parameter for >2 classes",
        "recommendation": "Use multi_class='ovr' or 'ovo' for multiclass AUC",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
    },
    {
        "name": "comparing_different_splits",
        "pattern": r"(test_size|random_state)\s*=[\s\S]{0,300}\1\s*=(?!\s*\k<1>)",
        "description": "Different random_state or test_size in comparisons",
        "why": "Comparing models on different splits is not fair",
        "recommendation": "Use same split (same random_state, test_size) for all models",
        "tier": IssueTier.RESULTS_QUESTIONABLE,
        "requires_context": True,
    },
]


LLM_DETECTION_PROMPT = """You are a machine learning expert auditing code for critical issues that could invalidate results.

Review this code for catastrophic patterns:

```{language}
{code}
```

File: {file_path}

Check for these issues:

**Data Leakage (tier1 - results invalid):**
- Preprocessing (normalization, scaling, feature selection) before train/test split
- Test data statistics influencing training
- Cross-validation with preprocessing outside folds
- Information from future time steps in temporal data

**Evaluation Errors (tier1 - results invalid):**
- Computing metrics on training data instead of test
- Wrong metric for task (e.g., accuracy for imbalanced data)
- Test set used during hyperparameter tuning

**Training Issues (tier1-2):**
- Model in wrong mode (train vs eval) during evaluation
- Gradient issues (not zeroing, detached tensors)
- Loss function mismatch for task

**Reproducibility (tier3):**
- No random seed setting
- Non-deterministic operations without flags

**Data Integrity (tier1-2):**
- Silent type coercion / precision loss
- NaN values not handled
- Incorrect tensor reshaping

For each issue found, provide a JSON object with:
- pattern_type: short identifier (e.g., "fit_before_split")
- category: "data_leakage" | "evaluation_error" | "training_issue" | "reproducibility" | "data_integrity"
- tier: "tier1" | "tier2" | "tier3" | "tier4"
- line_start: approximate line number (or 0 if unknown)
- line_end: approximate line number
- description: what the issue is
- why_catastrophic: why this invalidates or affects results
- recommendation: how to fix it

Return a JSON array of issues found. Return [] if no issues.
Return ONLY valid JSON, no other text."""


class CatastrophicDetector:
    """Detect catastrophic patterns in code."""

    def __init__(self, enable_phase4: bool = True):
        self.ai_provider = None
        self.enable_phase4 = enable_phase4  # Enable extended patterns

    async def detect_all(
        self,
        codebase: Codebase,
        categories: Optional[list[CatastrophicCategory]] = None,
        use_llm: bool = True,
        max_concurrent: int = 5,
    ) -> list[CatastrophicWarning]:
        """
        Detect all catastrophic patterns in codebase.

        Args:
            codebase: The codebase to analyze
            categories: Optional filter for specific categories
            use_llm: Whether to use LLM for additional detection
            max_concurrent: Maximum concurrent LLM calls

        Returns:
            List of CatastrophicWarning objects
        """
        if use_llm and self.ai_provider is None:
            self.ai_provider = get_ai_provider()

        warnings = []

        # Get relevant files
        relevant_files = self._get_relevant_files(codebase)

        # Pattern-based detection (fast, run first)
        for file in relevant_files:
            pattern_warnings = self._detect_patterns(file)
            warnings.extend(pattern_warnings)

        # LLM-based detection in parallel
        if use_llm:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def detect_with_limit(file):
                async with semaphore:
                    return await self._detect_with_llm(file)

            llm_tasks = [detect_with_limit(file) for file in relevant_files]
            llm_results = await asyncio.gather(*llm_tasks)

            for file_warnings in llm_results:
                warnings.extend(file_warnings)

        # Filter by category if specified
        if categories:
            warnings = [w for w in warnings if w.category in categories]

        # Deduplicate warnings
        warnings = self._deduplicate_warnings(warnings)

        return warnings

    def _detect_patterns(self, file: CodeFile) -> list[CatastrophicWarning]:
        """Detect issues using regex patterns."""
        warnings = []
        content = file.content
        content_lower = content.lower()

        # Build pattern list based on phases
        all_patterns = [
            # Phase 3: Core catastrophic patterns
            (LEAKAGE_PATTERNS, CatastrophicCategory.DATA_LEAKAGE),
            (EVALUATION_ERROR_PATTERNS, CatastrophicCategory.EVALUATION_ERROR),
            (TRAINING_ISSUE_PATTERNS, CatastrophicCategory.TRAINING_ISSUE),
        ]

        # Phase 4: Extended patterns
        if self.enable_phase4:
            all_patterns.extend([
                (REPRODUCIBILITY_PATTERNS, CatastrophicCategory.REPRODUCIBILITY),
                (DATA_INTEGRITY_PATTERNS, CatastrophicCategory.DATA_INTEGRITY),
                (STATISTICAL_ERROR_PATTERNS, CatastrophicCategory.STATISTICAL_ERROR),
                # Phase 7: Expanded pattern library
                (TIME_SERIES_PATTERNS, CatastrophicCategory.DATA_LEAKAGE),
                (DEEP_LEARNING_PATTERNS, CatastrophicCategory.TRAINING_ISSUE),
                (SKLEARN_PATTERNS, CatastrophicCategory.DATA_LEAKAGE),
                (METRIC_PATTERNS, CatastrophicCategory.EVALUATION_ERROR),
            ])

        for pattern_list, category in all_patterns:
            for pattern_def in pattern_list:
                detected = self._check_single_pattern(pattern_def, file, content, content_lower)
                if detected:
                    for warning in detected:
                        warning.category = category
                        warnings.append(warning)

        return warnings

    def _check_single_pattern(
        self,
        pattern_def: dict,
        file: CodeFile,
        content: str,
        content_lower: str,
    ) -> list[CatastrophicWarning]:
        """Check a single pattern definition against file content."""
        warnings = []

        # Skip patterns that need LLM analysis (context-dependent)
        if pattern_def.get("requires_context"):
            return []

        # Handle patterns that require absence of certain strings
        if "requires_absence" in pattern_def:
            # First check if the trigger pattern exists
            trigger_match = re.search(pattern_def["pattern"], content, re.IGNORECASE | re.DOTALL)
            if not trigger_match:
                return []

            # Check if any of the required strings are absent
            absence_list = pattern_def["requires_absence"]
            all_present = all(s.lower() in content_lower for s in absence_list)
            if all_present:
                return []  # All required strings present, no warning

            # Some required strings are missing - create warning
            if self._is_training_file(file):
                warnings.append(CatastrophicWarning(
                    id=str(uuid.uuid4()),
                    category=CatastrophicCategory.REPRODUCIBILITY,  # Will be overwritten
                    pattern_type=pattern_def["name"],
                    tier=pattern_def["tier"],
                    file_path=file.path,
                    relative_path=file.relative_path,
                    line_start=1,
                    line_end=min(50, file.line_count),
                    description=pattern_def["description"],
                    code_snippet=content[:500],
                    why_catastrophic=pattern_def["why"],
                    recommendation=pattern_def["recommendation"],
                ))
            return warnings

        # Normal pattern matching
        try:
            for match in re.finditer(pattern_def["pattern"], content, re.IGNORECASE | re.DOTALL):
                # Find line numbers
                start_pos = match.start()
                line_start = content[:start_pos].count("\n") + 1
                line_end = line_start + match.group().count("\n")

                # Get code snippet with context
                snippet_start = max(0, start_pos - 100)
                snippet_end = min(len(content), match.end() + 100)
                snippet = content[snippet_start:snippet_end]

                warnings.append(CatastrophicWarning(
                    id=str(uuid.uuid4()),
                    category=CatastrophicCategory.DATA_LEAKAGE,  # Will be overwritten
                    pattern_type=pattern_def["name"],
                    tier=pattern_def["tier"],
                    file_path=file.path,
                    relative_path=file.relative_path,
                    line_start=line_start,
                    line_end=line_end,
                    description=pattern_def["description"],
                    code_snippet=snippet,
                    why_catastrophic=pattern_def["why"],
                    recommendation=pattern_def["recommendation"],
                ))

        except re.error:
            # Skip malformed regex patterns
            pass

        return warnings

    async def _detect_with_llm(self, file: CodeFile) -> list[CatastrophicWarning]:
        """Detect issues using LLM analysis."""
        # Only analyze Python files for now
        if file.language != "python":
            return []

        # Skip very large files
        if file.line_count > 500:
            # Analyze in chunks
            return await self._detect_large_file(file)

        prompt = LLM_DETECTION_PROMPT.format(
            language=file.language,
            code=file.content,
            file_path=file.relative_path,
        )

        try:
            response = await self.ai_provider.complete(
                prompt=prompt,
            )

            return self._parse_llm_response(response, file)
        except Exception:
            return []

    async def _detect_large_file(self, file: CodeFile, max_concurrent: int = 3) -> list[CatastrophicWarning]:
        """Analyze large files in chunks (parallelized)."""
        lines = file.content.split("\n")
        chunk_size = 300
        overlap = 50

        # Build all chunks
        chunks = []
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = "\n".join(chunk_lines)
            chunks.append((i, chunk_content, len(chunk_lines)))

        # Process chunks in parallel
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_chunk(offset: int, content: str, num_lines: int):
            async with semaphore:
                prompt = LLM_DETECTION_PROMPT.format(
                    language=file.language,
                    code=content,
                    file_path=f"{file.relative_path} (lines {offset+1}-{offset+num_lines})",
                )
                try:
                    response = await self.ai_provider.complete(prompt=prompt)
                    return self._parse_llm_response(response, file, line_offset=offset)
                except Exception:
                    return []

        tasks = [analyze_chunk(offset, content, num_lines) for offset, content, num_lines in chunks]
        results = await asyncio.gather(*tasks)

        # Flatten results
        warnings = []
        for chunk_warnings in results:
            warnings.extend(chunk_warnings)

        return warnings

    def _parse_llm_response(
        self,
        response: str,
        file: CodeFile,
        line_offset: int = 0,
    ) -> list[CatastrophicWarning]:
        """Parse LLM response into warnings."""
        warnings = []

        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            if not isinstance(data, list):
                data = [data] if data else []

            for item in data:
                # Map category string to enum
                cat_str = item.get("category", "implementation_bug")
                try:
                    category = CatastrophicCategory(cat_str)
                except ValueError:
                    category = CatastrophicCategory.IMPLEMENTATION_BUG

                # Map tier string to enum
                tier_str = item.get("tier", "tier4")
                try:
                    tier = IssueTier(tier_str)
                except ValueError:
                    tier = IssueTier.MINOR_DISCREPANCY

                line_start = item.get("line_start", 1) + line_offset
                line_end = item.get("line_end", line_start) + line_offset

                # Extract code snippet
                snippet = self._extract_snippet(file.content, line_start, line_end)

                warnings.append(CatastrophicWarning(
                    id=str(uuid.uuid4()),
                    category=category,
                    pattern_type=item.get("pattern_type", "unknown"),
                    tier=tier,
                    file_path=file.path,
                    relative_path=file.relative_path,
                    line_start=line_start,
                    line_end=line_end,
                    description=item.get("description", ""),
                    code_snippet=snippet,
                    why_catastrophic=item.get("why_catastrophic", ""),
                    recommendation=item.get("recommendation", ""),
                ))

        except json.JSONDecodeError:
            pass

        return warnings

    def _extract_snippet(self, content: str, line_start: int, line_end: int) -> str:
        """Extract code snippet around given lines."""
        lines = content.split("\n")
        start = max(0, line_start - 3)
        end = min(len(lines), line_end + 3)
        return "\n".join(lines[start:end])

    def _get_relevant_files(self, codebase: Codebase) -> list[CodeFile]:
        """Get files that are likely to contain issues."""
        relevant = []
        relevant_keywords = [
            "train", "data", "model", "eval", "test", "valid",
            "preprocess", "transform", "split", "main", "run",
        ]

        for file in codebase.files:
            if file.language != "python":
                continue

            file_lower = file.relative_path.lower()
            if any(kw in file_lower for kw in relevant_keywords):
                relevant.append(file)
            elif "main" in file_lower or file_lower.endswith("run.py"):
                relevant.append(file)

        # If nothing found, include all Python files up to a limit
        if not relevant:
            relevant = [f for f in codebase.files if f.language == "python"][:15]

        return relevant

    def _is_training_file(self, file: CodeFile) -> bool:
        """Check if file is likely a training script."""
        keywords = ["train", "main", "run", "experiment"]
        return any(kw in file.relative_path.lower() for kw in keywords)

    def _deduplicate_warnings(
        self,
        warnings: list[CatastrophicWarning],
    ) -> list[CatastrophicWarning]:
        """Remove duplicate warnings."""
        seen = set()
        unique = []

        for w in warnings:
            # Create a key based on file, pattern, and approximate location
            key = (w.relative_path, w.pattern_type, w.line_start // 10)
            if key not in seen:
                seen.add(key)
                unique.append(w)

        return unique
