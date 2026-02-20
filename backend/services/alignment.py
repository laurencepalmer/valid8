import json
import re
from typing import Optional

from backend.config import get_settings
from backend.models.analysis import (
    CodeReference,
    HighlightAnalysisResponse,
    AlignmentCheckResponse,
    AlignmentIssue,
    PaperReference,
    CodeHighlightAnalysisResponse,
)
from backend.models.codebase import Codebase
from backend.models.paper import Paper
from backend.services.ai import get_ai_provider
from backend.services.embeddings import get_embedding_service
from backend.services.search import get_hybrid_searcher, get_reranker
from backend.services.state import app_state


HIGHLIGHT_SYSTEM_PROMPT = """You are an expert at connecting research papers to their code implementations.

Given highlighted text from a paper and numbered code snippets (each labeled with its file path), determine which snippets implement or relate to the paper text and explain **how** — referencing specific functions, classes, variables, or logic by name.

You MUST respond with ONLY valid JSON in this exact structure:
{"relevant_snippets": [{"index": 0, "relevance_score": 0.85, "explanation": "..."}], "summary": "..."}

Rules:
- "index" is the snippet number (0, 1, 2, …) — use the number, not the file name
- "relevance_score" is 0.0–1.0
- "explanation" must be specific: name the function/class/variable and describe what it does relative to the paper text. Never say "snippet_0" or "this snippet" — refer to code by its actual name (e.g. "The `compute_attention()` function implements the scaled dot-product attention described here")
- "summary" should be 1–2 sentences connecting the paper concept to the concrete code. Use real names from the code, not generic labels
- Only include snippets with a genuine connection (score >= 0.5). Omit irrelevant ones"""


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


CODE_TO_PAPER_SYSTEM_PROMPT = """You are an expert at connecting code implementations back to their research paper origins.

Given a code snippet and numbered paper sections, determine which sections describe or motivate the code and explain **how** — referencing specific concepts, equations, algorithms, or definitions from the paper by name.

You MUST respond with ONLY valid JSON in this exact structure:
{"relevant_sections": [{"index": 0, "relevance_score": 0.85, "explanation": "..."}], "summary": "..."}

Rules:
- "index" is the section number (0, 1, 2, …)
- "relevance_score" is 0.0–1.0
- "explanation" must be specific: name the paper concept/equation/algorithm and describe how the code implements or relates to it. Never say "section_0" or "this section" — refer to content by its actual name
- "summary" should be 1–2 sentences connecting the code to the paper concepts. Use real names from both the code and the paper
- Only include sections with a genuine connection (score >= 0.5). Omit irrelevant ones"""


HIGHLIGHT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_snippets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance_score": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["index", "relevance_score", "explanation"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["relevant_snippets", "summary"],
}

CODE_HIGHLIGHT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance_score": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["index", "relevance_score", "explanation"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["relevant_sections", "summary"],
}


# Maps code patterns to natural-language descriptions for paper search.
# Covers PyTorch, TensorFlow/Keras, JAX/Flax, NumPy/SciPy, scikit-learn,
# R, Julia, MATLAB, and common research-paper idioms.
CODE_OPERATION_MAP = {
    # ── Tensor / array reshaping ──────────────────────────────────────
    '.view(': 'reshape tensor',
    '.reshape(': 'reshape tensor',
    'np.reshape(': 'reshape array',
    'tf.reshape(': 'reshape tensor',
    'jnp.reshape(': 'reshape array',
    '.permute(': 'transpose dimensions rearrange',
    '.transpose(': 'transpose dimensions',
    'np.transpose(': 'transpose array',
    'tf.transpose(': 'transpose tensor',
    'jnp.transpose(': 'transpose array',
    '.contiguous(': 'contiguous memory layout',
    '.flatten(': 'flatten tensor',
    'np.ravel(': 'flatten array',
    'tf.expand_dims(': 'expand dimensions unsqueeze',
    '.unsqueeze(': 'expand dimensions unsqueeze',
    '.squeeze(': 'remove dimensions squeeze',
    'np.expand_dims(': 'expand dimensions',
    'np.squeeze(': 'remove dimensions',
    'torch.stack(': 'stack tensors',
    'torch.cat(': 'concatenate tensors',
    'np.concatenate(': 'concatenate arrays',
    'tf.concat(': 'concatenate tensors',
    'jnp.concatenate(': 'concatenate arrays',
    'np.stack(': 'stack arrays',
    'torch.split(': 'split tensor',
    'np.split(': 'split array',
    'torch.chunk(': 'chunk tensor split',
    'einops.rearrange': 'rearrange tensor dimensions',
    'einops.repeat': 'repeat tensor tile',
    'einops.reduce': 'reduce tensor aggregation',

    # ── Matrix / linear algebra ───────────────────────────────────────
    '@': 'matrix multiplication',
    'torch.matmul': 'matrix multiplication',
    'torch.bmm': 'batched matrix multiplication',
    'np.matmul(': 'matrix multiplication',
    'np.dot(': 'dot product matrix multiplication',
    'tf.matmul(': 'matrix multiplication',
    'jnp.matmul(': 'matrix multiplication',
    'jnp.dot(': 'dot product',
    'torch.einsum': 'Einstein summation contraction',
    'np.einsum(': 'Einstein summation contraction',
    'jnp.einsum(': 'Einstein summation contraction',
    'tf.einsum(': 'Einstein summation contraction',
    'torch.linalg.inv': 'matrix inverse',
    'np.linalg.inv(': 'matrix inverse',
    'torch.linalg.svd': 'singular value decomposition SVD',
    'np.linalg.svd(': 'singular value decomposition SVD',
    'torch.linalg.eig': 'eigenvalue decomposition',
    'np.linalg.eig(': 'eigenvalue decomposition eigenvectors',
    'np.linalg.eigh(': 'eigenvalue decomposition symmetric hermitian',
    'torch.linalg.det': 'determinant',
    'np.linalg.det(': 'determinant',
    'torch.linalg.norm': 'matrix norm vector norm',
    'np.linalg.norm(': 'norm',
    'torch.linalg.solve': 'solve linear system',
    'np.linalg.solve(': 'solve linear system',
    'np.linalg.lstsq(': 'least squares solution',
    'torch.linalg.cholesky': 'Cholesky decomposition',
    'np.linalg.cholesky(': 'Cholesky decomposition',
    'np.linalg.qr(': 'QR decomposition factorization',
    'torch.linalg.qr': 'QR decomposition factorization',
    'torch.trace(': 'matrix trace',
    'np.trace(': 'matrix trace',
    'np.outer(': 'outer product',
    'np.inner(': 'inner product',
    'np.kron(': 'Kronecker product',
    'torch.cross(': 'cross product',
    'np.cross(': 'cross product',
    'scipy.linalg.lu(': 'LU decomposition factorization',
    'scipy.linalg.schur(': 'Schur decomposition',
    'scipy.sparse': 'sparse matrix',

    # ── Activation functions ──────────────────────────────────────────
    '.softmax(': 'softmax normalization attention weights',
    'F.softmax': 'softmax normalization',
    'tf.nn.softmax': 'softmax normalization',
    'jax.nn.softmax': 'softmax normalization',
    'nn.ReLU': 'ReLU activation rectified linear unit',
    'F.relu': 'ReLU activation',
    'tf.nn.relu': 'ReLU activation',
    'jax.nn.relu': 'ReLU activation',
    'nn.GELU': 'GELU activation Gaussian error linear unit',
    'F.gelu': 'GELU activation',
    'jax.nn.gelu': 'GELU activation',
    'nn.SiLU': 'SiLU Swish activation',
    'F.silu': 'SiLU Swish activation',
    'nn.Sigmoid': 'sigmoid activation',
    'torch.sigmoid': 'sigmoid activation',
    'nn.Tanh': 'tanh activation hyperbolic tangent',
    'nn.LeakyReLU': 'leaky ReLU activation',
    'nn.PReLU': 'parametric ReLU activation',
    'nn.ELU': 'ELU activation exponential linear unit',
    'nn.Mish': 'Mish activation',
    'nn.Softplus': 'softplus activation',
    'nn.Hardswish': 'hard Swish activation',
    'F.log_softmax': 'log softmax',
    'tf.nn.leaky_relu': 'leaky ReLU activation',
    'tf.nn.elu': 'ELU activation',
    'tf.nn.selu': 'SELU scaled exponential linear unit',

    # ── Neural network layers ─────────────────────────────────────────
    'nn.Linear': 'linear projection fully connected layer',
    'nn.Conv1d': 'convolution 1d temporal',
    'nn.Conv2d': 'convolution 2d spatial',
    'nn.Conv3d': 'convolution 3d volumetric',
    'nn.ConvTranspose2d': 'transposed convolution deconvolution upsampling',
    'nn.ConvTranspose1d': 'transposed convolution deconvolution',
    'nn.DepthwiseConv': 'depthwise convolution',
    'nn.LayerNorm': 'layer normalization',
    'nn.BatchNorm': 'batch normalization',
    'nn.GroupNorm': 'group normalization',
    'nn.InstanceNorm': 'instance normalization',
    'nn.RMSNorm': 'root mean square normalization RMSNorm',
    'nn.Dropout': 'dropout regularization',
    'nn.Embedding': 'embedding lookup table',
    'nn.MultiheadAttention': 'multi-head attention',
    'nn.TransformerEncoder': 'transformer encoder',
    'nn.TransformerDecoder': 'transformer decoder',
    'nn.LSTM': 'long short-term memory LSTM recurrent',
    'nn.GRU': 'gated recurrent unit GRU recurrent',
    'nn.RNN': 'recurrent neural network RNN',
    'nn.MaxPool2d': 'max pooling',
    'nn.AvgPool2d': 'average pooling',
    'nn.AdaptiveAvgPool': 'global average pooling',
    'nn.Upsample': 'upsampling interpolation',
    'F.interpolate': 'interpolation upsampling bilinear',
    'nn.PixelShuffle': 'pixel shuffle sub-pixel convolution',
    'nn.Sequential': 'sequential model',
    'nn.ModuleList': 'module list',
    '.norm(': 'normalization',

    # ── TensorFlow / Keras layers ─────────────────────────────────────
    'tf.keras.layers.Dense': 'fully connected dense layer',
    'keras.layers.Dense': 'fully connected dense layer',
    'layers.Dense': 'fully connected dense layer',
    'tf.keras.layers.Conv2D': 'convolution 2d',
    'keras.layers.Conv2D': 'convolution 2d',
    'layers.Conv2D': 'convolution 2d',
    'tf.keras.layers.Conv1D': 'convolution 1d temporal',
    'tf.keras.layers.LSTM': 'LSTM recurrent',
    'keras.layers.LSTM': 'LSTM recurrent',
    'tf.keras.layers.GRU': 'GRU recurrent',
    'tf.keras.layers.Embedding': 'embedding layer',
    'tf.keras.layers.BatchNormalization': 'batch normalization',
    'tf.keras.layers.LayerNormalization': 'layer normalization',
    'tf.keras.layers.Dropout': 'dropout regularization',
    'tf.keras.layers.MultiHeadAttention': 'multi-head attention',
    'tf.keras.layers.Attention': 'attention layer',
    'tf.keras.layers.GlobalAveragePooling': 'global average pooling',
    'tf.keras.layers.MaxPooling2D': 'max pooling',
    'tf.keras.layers.Flatten': 'flatten layer',
    'tf.keras.layers.Concatenate': 'concatenation layer',
    'tf.keras.layers.Add': 'residual addition skip connection',
    'tf.keras.Model': 'model definition',
    'tf.GradientTape': 'automatic differentiation gradient computation',

    # ── JAX / Flax / Haiku ────────────────────────────────────────────
    'jax.grad(': 'gradient computation automatic differentiation',
    'jax.jit(': 'just-in-time compilation',
    'jax.vmap(': 'vectorized map batching',
    'jax.pmap(': 'parallel map distributed computation',
    'jax.lax.scan': 'sequential scan loop',
    'jax.lax.conv': 'convolution',
    'jax.random': 'random number generation',
    'flax.linen': 'Flax neural network module',
    'nn.Dense': 'fully connected dense layer',
    'nn.Conv': 'convolution layer',
    'hk.Linear': 'linear layer Haiku',
    'hk.Conv2D': 'convolution 2d Haiku',
    'hk.MultiHeadAttention': 'multi-head attention Haiku',
    'hk.BatchNorm': 'batch normalization Haiku',
    'hk.LayerNorm': 'layer normalization Haiku',
    'optax.adam': 'Adam optimizer',
    'optax.sgd': 'SGD optimizer stochastic gradient descent',
    'optax.chain': 'optimizer chain',
    'optax.clip_by_global_norm': 'gradient clipping',

    # ── Attention / transformer patterns ──────────────────────────────
    'attention': 'attention mechanism',
    'self_attention': 'self-attention',
    'cross_attention': 'cross-attention',
    'multi_head': 'multi-head attention',
    'scaled_dot_product': 'scaled dot-product attention',
    'qkv': 'query key value projection',
    'q_proj': 'query projection',
    'k_proj': 'key projection',
    'v_proj': 'value projection',
    'relative_position': 'relative position bias encoding',
    'rotary': 'rotary position embedding RoPE',
    'alibi': 'ALiBi attention linear bias',
    'flash_attn': 'flash attention efficient attention',
    'xformers': 'memory efficient attention',
    'mask': 'attention mask',
    'causal_mask': 'causal mask autoregressive',
    'cls_token': 'class token CLS',
    'pos_embed': 'positional embedding encoding',
    'patch_embed': 'patch embedding',
    'window': 'window partition',
    'roll(': 'cyclic shift roll',
    'torch.roll': 'cyclic shift roll',
    'feed_forward': 'feed-forward network FFN MLP',
    'ffn': 'feed-forward network',
    'residual': 'residual connection skip connection',
    'shortcut': 'skip connection residual',
    'layer_scale': 'layer scale',
    'stochastic_depth': 'stochastic depth drop path',
    'drop_path': 'drop path stochastic depth',
    'DropPath': 'drop path stochastic depth',

    # ── Loss functions ────────────────────────────────────────────────
    'cross_entropy': 'cross-entropy loss',
    'CrossEntropyLoss': 'cross-entropy loss classification',
    'F.cross_entropy': 'cross-entropy loss',
    'nn.CrossEntropyLoss': 'cross-entropy loss',
    'nn.BCELoss': 'binary cross-entropy loss',
    'F.binary_cross_entropy': 'binary cross-entropy loss',
    'nn.MSELoss': 'mean squared error loss regression',
    'F.mse_loss': 'mean squared error loss',
    'nn.L1Loss': 'L1 loss mean absolute error',
    'nn.SmoothL1Loss': 'smooth L1 loss Huber loss',
    'nn.KLDivLoss': 'KL divergence loss',
    'F.kl_div': 'KL divergence Kullback-Leibler',
    'nn.NLLLoss': 'negative log likelihood loss',
    'nn.TripletMarginLoss': 'triplet margin loss metric learning',
    'nn.CosineEmbeddingLoss': 'cosine embedding loss',
    'nn.HingeEmbeddingLoss': 'hinge loss',
    'nn.CTCLoss': 'CTC loss connectionist temporal classification',
    'focal_loss': 'focal loss class imbalance',
    'contrastive_loss': 'contrastive loss',
    'InfoNCE': 'InfoNCE loss contrastive',
    'dice_loss': 'Dice loss segmentation',
    'iou_loss': 'IoU loss intersection over union',
    'tf.keras.losses': 'loss function',
    'tf.nn.sigmoid_cross_entropy': 'sigmoid cross-entropy loss',
    'tf.nn.softmax_cross_entropy': 'softmax cross-entropy loss',

    # ── Optimizers ────────────────────────────────────────────────────
    'optim.Adam': 'Adam optimizer',
    'optim.AdamW': 'AdamW optimizer weight decay',
    'optim.SGD': 'stochastic gradient descent SGD',
    'optim.RMSprop': 'RMSprop optimizer',
    'optim.Adagrad': 'Adagrad optimizer adaptive gradient',
    'optim.LBFGS': 'L-BFGS optimizer quasi-Newton',
    'torch.optim.lr_scheduler': 'learning rate scheduler',
    'CosineAnnealingLR': 'cosine annealing learning rate schedule',
    'StepLR': 'step learning rate decay',
    'ReduceLROnPlateau': 'reduce learning rate on plateau',
    'WarmupCosine': 'warmup cosine learning rate schedule',
    'OneCycleLR': 'one cycle learning rate policy',
    'tf.keras.optimizers.Adam': 'Adam optimizer',
    'tf.keras.optimizers.SGD': 'SGD optimizer',

    # ── Regularization / normalization techniques ─────────────────────
    'weight_decay': 'weight decay L2 regularization',
    'label_smoothing': 'label smoothing regularization',
    'mixup': 'mixup data augmentation',
    'cutmix': 'CutMix data augmentation',
    'cutout': 'Cutout data augmentation',
    'spectral_norm': 'spectral normalization',
    'gradient_penalty': 'gradient penalty regularization',
    'clip_grad_norm': 'gradient clipping',
    'nn.utils.clip_grad_norm': 'gradient clipping norm',
    'nn.utils.clip_grad_value': 'gradient clipping value',

    # ── Pooling / aggregation ─────────────────────────────────────────
    'global_avg_pool': 'global average pooling',
    'max_pool': 'max pooling',
    'avg_pool': 'average pooling',
    'F.adaptive_avg_pool': 'adaptive average pooling',
    'topk(': 'top-k selection',
    'torch.topk': 'top-k selection',
    'torch.argmax': 'argmax',
    'torch.argmin': 'argmin',
    'torch.mean(': 'mean average',
    'torch.sum(': 'summation',
    'torch.var(': 'variance',
    'torch.std(': 'standard deviation',
    'torch.cumsum(': 'cumulative sum',
    'torch.cumprod(': 'cumulative product',

    # ── Probability / distributions ───────────────────────────────────
    'torch.distributions': 'probability distribution sampling',
    'Normal(': 'normal distribution Gaussian',
    'MultivariateNormal': 'multivariate normal Gaussian distribution',
    'Categorical(': 'categorical distribution sampling',
    'Bernoulli(': 'Bernoulli distribution',
    'Uniform(': 'uniform distribution',
    'Beta(': 'Beta distribution',
    'Dirichlet(': 'Dirichlet distribution',
    'Poisson(': 'Poisson distribution',
    'kl_divergence': 'KL divergence Kullback-Leibler',
    '.log_prob(': 'log probability likelihood',
    '.rsample(': 'reparameterized sampling reparameterization trick',
    '.sample(': 'sampling random',
    'Gumbel': 'Gumbel-Softmax sampling',
    'gumbel_softmax': 'Gumbel-Softmax straight-through estimator',
    'tf.random.normal': 'normal distribution sampling',
    'tf.random.categorical': 'categorical sampling',

    # ── NumPy / SciPy core ────────────────────────────────────────────
    'np.array(': 'array creation',
    'np.zeros(': 'zero array initialization',
    'np.ones(': 'ones array initialization',
    'np.eye(': 'identity matrix',
    'np.random': 'random number generation',
    'np.mean(': 'mean average',
    'np.std(': 'standard deviation',
    'np.var(': 'variance',
    'np.sum(': 'summation',
    'np.prod(': 'product',
    'np.cumsum(': 'cumulative sum',
    'np.clip(': 'clipping values',
    'np.where(': 'conditional selection',
    'np.argmax(': 'argmax index of maximum',
    'np.argmin(': 'argmin index of minimum',
    'np.argsort(': 'argument sort indices',
    'np.sort(': 'sorting',
    'np.unique(': 'unique values',
    'np.interp(': 'linear interpolation',
    'np.histogram(': 'histogram binning',
    'np.percentile(': 'percentile quantile',
    'np.corrcoef(': 'correlation coefficient Pearson',
    'np.cov(': 'covariance matrix',
    'np.convolve(': 'convolution 1d',
    'np.fft': 'Fourier transform FFT frequency domain',
    'scipy.fft': 'Fourier transform FFT frequency domain',
    'np.log(': 'logarithm',
    'np.exp(': 'exponential',
    'np.sqrt(': 'square root',
    'np.abs(': 'absolute value',
    'np.sign(': 'sign function',
    'np.sin(': 'sine trigonometric',
    'np.cos(': 'cosine trigonometric',
    'np.pi': 'pi constant',
    'np.inf': 'infinity',
    'np.nan': 'not a number missing value',

    # ── SciPy scientific computing ────────────────────────────────────
    'scipy.optimize.minimize': 'optimization minimization',
    'scipy.optimize.curve_fit': 'curve fitting nonlinear regression',
    'scipy.optimize.least_squares': 'least squares optimization',
    'scipy.optimize.linear_sum_assignment': 'linear assignment Hungarian algorithm',
    'scipy.optimize.linprog': 'linear programming optimization',
    'scipy.optimize.differential_evolution': 'differential evolution optimization',
    'scipy.integrate.odeint': 'ODE ordinary differential equation integration',
    'scipy.integrate.solve_ivp': 'initial value problem ODE solver',
    'scipy.integrate.quad': 'numerical integration quadrature',
    'scipy.interpolate': 'interpolation spline',
    'scipy.interpolate.interp1d': 'interpolation 1d',
    'scipy.interpolate.griddata': 'grid interpolation scattered data',
    'scipy.signal.convolve': 'signal convolution filtering',
    'scipy.signal.fftconvolve': 'FFT convolution filtering',
    'scipy.signal.butter': 'Butterworth filter design',
    'scipy.signal.savgol_filter': 'Savitzky-Golay smoothing filter',
    'scipy.signal.find_peaks': 'peak detection',
    'scipy.signal.stft': 'short-time Fourier transform spectrogram',
    'scipy.signal.welch': 'power spectral density Welch',
    'scipy.ndimage': 'image processing filtering morphology',
    'scipy.spatial.distance': 'distance metric pairwise',
    'scipy.spatial.KDTree': 'KD-tree nearest neighbor spatial index',
    'scipy.spatial.Delaunay': 'Delaunay triangulation',
    'scipy.spatial.ConvexHull': 'convex hull',
    'scipy.spatial.Voronoi': 'Voronoi tessellation',
    'scipy.cluster': 'clustering',
    'scipy.special': 'special functions',
    'scipy.special.gamma': 'gamma function',
    'scipy.special.softmax': 'softmax',
    'scipy.special.expit': 'sigmoid logistic function',
    'scipy.special.logsumexp': 'log-sum-exp stable computation',

    # ── SciPy statistics ──────────────────────────────────────────────
    'scipy.stats': 'statistical test distribution',
    'scipy.stats.norm': 'normal distribution Gaussian',
    'scipy.stats.ttest': 't-test hypothesis testing',
    'scipy.stats.chi2': 'chi-squared test',
    'scipy.stats.ks_2samp': 'Kolmogorov-Smirnov test',
    'scipy.stats.pearsonr': 'Pearson correlation coefficient',
    'scipy.stats.spearmanr': 'Spearman rank correlation',
    'scipy.stats.mannwhitneyu': 'Mann-Whitney U test',
    'scipy.stats.wilcoxon': 'Wilcoxon signed-rank test',
    'scipy.stats.fisher_exact': 'Fisher exact test',
    'scipy.stats.anova': 'ANOVA analysis of variance',
    'scipy.stats.entropy': 'Shannon entropy',

    # ── scikit-learn machine learning ─────────────────────────────────
    'sklearn.linear_model': 'linear model regression',
    'LinearRegression': 'linear regression',
    'LogisticRegression': 'logistic regression classification',
    'Ridge(': 'Ridge regression L2 regularization',
    'Lasso(': 'Lasso regression L1 regularization',
    'ElasticNet': 'Elastic Net regularization',
    'sklearn.svm': 'support vector machine SVM',
    'SVC(': 'support vector classifier SVM',
    'SVR(': 'support vector regression',
    'sklearn.tree': 'decision tree',
    'DecisionTreeClassifier': 'decision tree classification',
    'RandomForestClassifier': 'random forest ensemble',
    'RandomForestRegressor': 'random forest regression ensemble',
    'GradientBoostingClassifier': 'gradient boosting ensemble',
    'GradientBoostingRegressor': 'gradient boosting regression',
    'AdaBoostClassifier': 'AdaBoost boosting ensemble',
    'sklearn.ensemble': 'ensemble method',
    'VotingClassifier': 'voting ensemble',
    'BaggingClassifier': 'bagging bootstrap ensemble',
    'sklearn.neighbors': 'nearest neighbors',
    'KNeighborsClassifier': 'k-nearest neighbors KNN',
    'sklearn.cluster.KMeans': 'k-means clustering',
    'KMeans(': 'k-means clustering',
    'DBSCAN(': 'DBSCAN density clustering',
    'AgglomerativeClustering': 'agglomerative hierarchical clustering',
    'SpectralClustering': 'spectral clustering',
    'GaussianMixture': 'Gaussian mixture model GMM expectation maximization',
    'sklearn.decomposition.PCA': 'principal component analysis PCA dimensionality reduction',
    'PCA(': 'principal component analysis PCA',
    'sklearn.decomposition.NMF': 'non-negative matrix factorization NMF',
    'TruncatedSVD': 'truncated SVD dimensionality reduction',
    'TSNE(': 't-SNE dimensionality reduction visualization',
    'UMAP(': 'UMAP dimensionality reduction manifold',
    'sklearn.manifold': 'manifold learning dimensionality reduction',
    'sklearn.preprocessing': 'data preprocessing scaling',
    'StandardScaler': 'standardization z-score normalization',
    'MinMaxScaler': 'min-max normalization scaling',
    'LabelEncoder': 'label encoding categorical',
    'OneHotEncoder': 'one-hot encoding categorical',
    'sklearn.feature_extraction': 'feature extraction',
    'TfidfVectorizer': 'TF-IDF text feature extraction',
    'CountVectorizer': 'bag-of-words text feature extraction',
    'sklearn.model_selection': 'model selection',
    'cross_val_score': 'cross-validation evaluation',
    'GridSearchCV': 'grid search hyperparameter tuning',
    'RandomizedSearchCV': 'randomized search hyperparameter tuning',
    'train_test_split': 'train test split data partitioning',
    'StratifiedKFold': 'stratified k-fold cross-validation',
    'sklearn.metrics': 'evaluation metrics',
    'accuracy_score': 'accuracy metric classification',
    'precision_score': 'precision metric',
    'recall_score': 'recall metric sensitivity',
    'f1_score': 'F1 score metric',
    'roc_auc_score': 'ROC AUC metric',
    'confusion_matrix': 'confusion matrix classification',
    'mean_squared_error': 'mean squared error MSE',
    'r2_score': 'R-squared coefficient of determination',
    'silhouette_score': 'silhouette score clustering',
    'sklearn.pipeline': 'pipeline workflow',

    # ── XGBoost / LightGBM / CatBoost ────────────────────────────────
    'xgboost': 'XGBoost gradient boosting',
    'XGBClassifier': 'XGBoost classifier gradient boosting',
    'XGBRegressor': 'XGBoost regressor gradient boosting',
    'lightgbm': 'LightGBM gradient boosting',
    'LGBMClassifier': 'LightGBM classifier gradient boosting',
    'CatBoostClassifier': 'CatBoost categorical gradient boosting',

    # ── Computer vision operations ────────────────────────────────────
    'torchvision.transforms': 'image transformation augmentation',
    'torchvision.models': 'pretrained model backbone',
    'ResNet': 'ResNet residual network',
    'VGG': 'VGG network',
    'EfficientNet': 'EfficientNet',
    'DenseNet': 'DenseNet dense connection',
    'MobileNet': 'MobileNet lightweight',
    'ViT': 'Vision Transformer ViT',
    'cv2.imread': 'image reading loading',
    'cv2.resize': 'image resizing',
    'cv2.cvtColor': 'color space conversion',
    'cv2.GaussianBlur': 'Gaussian blur smoothing',
    'cv2.Canny': 'Canny edge detection',
    'cv2.threshold': 'image thresholding binarization',
    'cv2.findContours': 'contour detection',
    'cv2.HoughLines': 'Hough transform line detection',
    'cv2.morphologyEx': 'morphological operations',
    'cv2.warpAffine': 'affine transformation',
    'cv2.warpPerspective': 'perspective transformation homography',
    'nms(': 'non-maximum suppression NMS',
    'non_max_suppression': 'non-maximum suppression NMS',
    'roi_pool': 'region of interest pooling RoI',
    'roi_align': 'region of interest alignment RoI',
    'anchor': 'anchor boxes object detection',
    'bbox': 'bounding box',
    'iou(': 'intersection over union IoU',
    'FPN': 'feature pyramid network FPN',
    'deformable_conv': 'deformable convolution',
    'depthwise_conv': 'depthwise separable convolution',

    # ── NLP / text processing ─────────────────────────────────────────
    'transformers.AutoModel': 'pretrained transformer model',
    'transformers.AutoTokenizer': 'tokenizer',
    'BertModel': 'BERT bidirectional encoder representations',
    'GPT2': 'GPT-2 generative pretrained transformer',
    'T5': 'T5 text-to-text transformer',
    'tokenizer.encode': 'text tokenization encoding',
    'tokenizer.decode': 'text decoding',
    'word2vec': 'Word2Vec word embedding',
    'GloVe': 'GloVe global vectors word embedding',
    'beam_search': 'beam search decoding',
    'greedy_decode': 'greedy decoding',
    'nucleus_sampling': 'nucleus sampling top-p',
    'top_k_sampling': 'top-k sampling',
    'temperature_sampling': 'temperature sampling',
    'bleu_score': 'BLEU score machine translation',
    'rouge_score': 'ROUGE score summarization',
    'perplexity': 'perplexity language model',
    'BPE': 'byte pair encoding tokenization',
    'SentencePiece': 'SentencePiece tokenization',

    # ── Generative models ─────────────────────────────────────────────
    'Generator(': 'generator network GAN',
    'Discriminator(': 'discriminator network GAN',
    'VAE': 'variational autoencoder VAE',
    'Encoder(': 'encoder network',
    'Decoder(': 'decoder network',
    'reparameterize': 'reparameterization trick variational',
    'latent': 'latent space representation',
    'diffusion': 'diffusion model denoising',
    'noise_schedule': 'noise schedule diffusion',
    'denoise': 'denoising diffusion',
    'UNet': 'U-Net encoder-decoder skip connection',
    'flow': 'normalizing flow',

    # ── GNN / graph methods ───────────────────────────────────────────
    'torch_geometric': 'graph neural network GNN',
    'GCNConv': 'graph convolutional network GCN',
    'GATConv': 'graph attention network GAT',
    'SAGEConv': 'GraphSAGE graph sampling aggregation',
    'MessagePassing': 'message passing graph',
    'edge_index': 'edge index graph adjacency',
    'adjacency_matrix': 'adjacency matrix graph',
    'graph_conv': 'graph convolution',
    'node_embedding': 'node embedding graph',
    'networkx': 'graph analysis NetworkX',
    'nx.Graph': 'graph creation NetworkX',

    # ── Reinforcement learning ────────────────────────────────────────
    'gym.make': 'environment creation reinforcement learning',
    'gymnasium.make': 'environment creation reinforcement learning',
    'env.step': 'environment step action reward',
    'env.reset': 'environment reset',
    'replay_buffer': 'experience replay buffer',
    'ReplayBuffer': 'experience replay buffer',
    'policy_network': 'policy network actor',
    'value_network': 'value network critic',
    'actor_critic': 'actor-critic method',
    'advantage': 'advantage estimation',
    'discount_factor': 'discount factor gamma',
    'epsilon_greedy': 'epsilon-greedy exploration',
    'PPO': 'proximal policy optimization PPO',
    'DQN': 'deep Q-network DQN',
    'A2C': 'advantage actor-critic A2C',
    'SAC': 'soft actor-critic SAC',
    'TD3': 'twin delayed DDPG TD3',
    'DDPG': 'deep deterministic policy gradient DDPG',
    'MCTS': 'Monte Carlo tree search MCTS',
    'reward_shaping': 'reward shaping',

    # ── Bayesian / probabilistic programming ──────────────────────────
    'pymc.Model': 'Bayesian model probabilistic programming',
    'pymc.sample': 'MCMC sampling Bayesian inference',
    'pymc.Normal': 'normal prior Bayesian',
    'pymc.HalfNormal': 'half-normal prior',
    'pymc.Uniform': 'uniform prior',
    'pymc.Deterministic': 'deterministic variable',
    'numpyro': 'NumPyro probabilistic programming',
    'numpyro.sample': 'sampling probabilistic',
    'stan_model': 'Stan Bayesian model',
    'MCMC(': 'Markov chain Monte Carlo sampling',
    'NUTS(': 'no-U-turn sampler NUTS Hamiltonian Monte Carlo',
    'HMC(': 'Hamiltonian Monte Carlo',
    'variational_inference': 'variational inference',
    'ELBO': 'evidence lower bound ELBO variational',
    'posterior': 'posterior distribution Bayesian',
    'prior': 'prior distribution Bayesian',
    'likelihood': 'likelihood function',
    'marginal_likelihood': 'marginal likelihood model evidence',
    'bayes_factor': 'Bayes factor model comparison',

    # ── Gaussian processes ────────────────────────────────────────────
    'GaussianProcess': 'Gaussian process GP regression',
    'gpytorch': 'Gaussian process PyTorch',
    'GPyTorch': 'Gaussian process PyTorch',
    'RBFKernel': 'radial basis function kernel squared exponential',
    'MaternKernel': 'Matern kernel covariance',
    'ScaleKernel': 'scaled kernel',
    'kernel_matrix': 'kernel matrix covariance',
    'acquisition_function': 'acquisition function Bayesian optimization',
    'expected_improvement': 'expected improvement Bayesian optimization',

    # ── Time series ───────────────────────────────────────────────────
    'ARIMA': 'ARIMA autoregressive integrated moving average',
    'SARIMA': 'seasonal ARIMA',
    'exponential_smoothing': 'exponential smoothing forecasting',
    'autocorrelation': 'autocorrelation ACF',
    'partial_autocorrelation': 'partial autocorrelation PACF',
    'Prophet': 'Prophet time series forecasting',
    'seasonal_decompose': 'seasonal decomposition',
    'differencing': 'differencing stationarity',
    'statsmodels': 'statistical modeling time series',
    'pmdarima': 'auto ARIMA time series',

    # ── Differential equations ────────────────────────────────────────
    'odeint': 'ODE ordinary differential equation integration',
    'solve_ivp': 'initial value problem ODE solver',
    'torchdiffeq': 'neural ODE differential equation',
    'NeuralODE': 'neural ordinary differential equation',
    'adjoint': 'adjoint method backward integration',
    'euler_step': 'Euler method numerical integration',
    'runge_kutta': 'Runge-Kutta method numerical integration',
    'RK4': 'Runge-Kutta fourth order',
    'finite_difference': 'finite difference method',
    'finite_element': 'finite element method FEM',
    'pde_solve': 'partial differential equation PDE solver',

    # ── R language ────────────────────────────────────────────────────
    # Base R
    'matrix(': 'matrix creation',
    'data.frame(': 'data frame creation',
    'as.matrix(': 'convert to matrix',
    'apply(': 'apply function across array margins',
    'sapply(': 'apply function simplify',
    'lapply(': 'apply function list',
    'tapply(': 'apply function grouped',
    'mapply(': 'multivariate apply',
    'Reduce(': 'reduce fold accumulate',
    'which(': 'index selection conditional',
    'ifelse(': 'conditional selection vectorized',
    'subset(': 'data subset selection',
    'merge(': 'data merge join',
    'aggregate(': 'data aggregation grouped',
    'lm(': 'linear model regression',
    'glm(': 'generalized linear model regression',
    'nls(': 'nonlinear least squares regression',
    'aov(': 'analysis of variance ANOVA',
    'anova(': 'analysis of variance ANOVA',
    't.test(': 't-test hypothesis testing',
    'chisq.test(': 'chi-squared test',
    'wilcox.test(': 'Wilcoxon test nonparametric',
    'cor.test(': 'correlation test',
    'fisher.test(': 'Fisher exact test',
    'shapiro.test(': 'Shapiro-Wilk normality test',
    'ks.test(': 'Kolmogorov-Smirnov test',
    'p.adjust(': 'p-value adjustment multiple testing',
    'prop.test(': 'proportion test',
    'var.test(': 'variance test F-test',
    'mcnemar.test(': 'McNemar test paired categorical',
    'summary(': 'model summary statistics',
    'predict(': 'prediction model',
    'fitted(': 'fitted values model',
    'residuals(': 'residuals model diagnostics',
    'confint(': 'confidence interval',
    'coef(': 'coefficients model parameters',
    'optim(': 'optimization minimization',
    'nlminb(': 'nonlinear minimization optimization',
    'integrate(': 'numerical integration quadrature',
    'uniroot(': 'root finding',
    'convolve(': 'convolution',
    'filter(': 'filtering time series',
    'spectrum(': 'spectral analysis',
    'acf(': 'autocorrelation function',
    'pacf(': 'partial autocorrelation function',
    'arima(': 'ARIMA time series model',
    'stl(': 'seasonal decomposition STL',
    'kmeans(': 'k-means clustering',
    'hclust(': 'hierarchical clustering',
    'cutree(': 'cut dendrogram tree clustering',
    'dist(': 'distance matrix pairwise',
    'prcomp(': 'principal component analysis PCA',
    'princomp(': 'principal component analysis PCA',
    'factanal(': 'factor analysis',
    'cmdscale(': 'multidimensional scaling MDS',
    'density(': 'kernel density estimation',
    'ecdf(': 'empirical cumulative distribution function',
    'quantile(': 'quantile percentile',
    'sample(': 'random sampling',
    'rnorm(': 'random normal distribution',
    'dnorm(': 'normal density function',
    'pnorm(': 'normal cumulative distribution',
    'qnorm(': 'normal quantile function',
    'rbinom(': 'random binomial distribution',
    'rpois(': 'random Poisson distribution',
    'rgamma(': 'random Gamma distribution',
    'rbeta(': 'random Beta distribution',
    'mvrnorm(': 'multivariate normal distribution',
    'Sys.time(': 'timing measurement',
    'set.seed(': 'random seed reproducibility',
    # R tidyverse / dplyr / ggplot2
    '%>%': 'pipe operator chaining',
    '|>': 'pipe operator',
    'dplyr::filter': 'data filtering rows',
    'dplyr::select': 'data column selection',
    'dplyr::mutate': 'data column creation transformation',
    'dplyr::summarise': 'data summarization aggregation',
    'dplyr::group_by': 'data grouping',
    'dplyr::arrange': 'data sorting ordering',
    'dplyr::join': 'data join merge',
    'dplyr::left_join': 'left join merge',
    'dplyr::inner_join': 'inner join merge',
    'tidyr::pivot_longer': 'data reshaping wide to long',
    'tidyr::pivot_wider': 'data reshaping long to wide',
    'tidyr::gather': 'data reshaping gathering',
    'tidyr::spread': 'data reshaping spreading',
    'tidyr::separate': 'data string splitting',
    'tidyr::unnest': 'data unnesting expanding',
    'purrr::map': 'functional mapping iteration',
    'stringr::str_detect': 'string pattern matching',
    'stringr::str_extract': 'string extraction regex',
    'forcats::fct_reorder': 'factor reordering',
    'ggplot(': 'data visualization plot',
    'geom_point': 'scatter plot visualization',
    'geom_line': 'line plot visualization',
    'geom_bar': 'bar plot visualization',
    'geom_histogram': 'histogram visualization',
    'geom_boxplot': 'box plot visualization',
    'geom_density': 'density plot visualization',
    'geom_smooth': 'smooth trend line regression',
    'facet_wrap': 'faceted multi-panel plot',
    'facet_grid': 'faceted grid plot',
    # R statistics packages
    'lme4::lmer': 'linear mixed-effects model',
    'lme4::glmer': 'generalized linear mixed-effects model',
    'nlme::lme': 'linear mixed-effects model',
    'survival::coxph': 'Cox proportional hazards survival',
    'survival::survfit': 'survival curve Kaplan-Meier',
    'survival::Surv': 'survival object',
    'mgcv::gam': 'generalized additive model GAM',
    'mgcv::bam': 'generalized additive model large data',
    'glmnet(': 'regularized regression Lasso Ridge elastic net',
    'caret::train': 'model training machine learning',
    'randomForest(': 'random forest ensemble',
    'gbm(': 'gradient boosting machine',
    'e1071::svm': 'support vector machine SVM',
    'nnet::nnet': 'neural network single layer',
    'rpart(': 'recursive partitioning decision tree',
    'brms::brm': 'Bayesian regression model',
    'rstanarm::stan_glm': 'Bayesian generalized linear model',
    'MCMCpack': 'MCMC Bayesian inference',
    'pscl::zeroinfl': 'zero-inflated model count data',
    'MASS::glm.nb': 'negative binomial regression',
    'MASS::lda': 'linear discriminant analysis',
    'MASS::qda': 'quadratic discriminant analysis',
    'boot::boot': 'bootstrap resampling',
    'car::Anova': 'type II III ANOVA',
    'car::vif': 'variance inflation factor multicollinearity',
    'lavaan::sem': 'structural equation modeling SEM',
    'lavaan::cfa': 'confirmatory factor analysis',
    'psych::fa': 'exploratory factor analysis',
    'psych::alpha': 'Cronbach alpha reliability',
    'mice::mice': 'multiple imputation missing data',
    'Amelia::amelia': 'multiple imputation missing data',
    'pROC::roc': 'ROC curve analysis',
    'ranger(': 'fast random forest',
    'xgboost::xgb.train': 'XGBoost gradient boosting',
    'lightgbm::lgb.train': 'LightGBM gradient boosting',
    'keras::': 'Keras deep learning R',
    'torch::': 'PyTorch R interface',
    'tensorflow::': 'TensorFlow R interface',
    'Rtsne::Rtsne': 't-SNE dimensionality reduction',
    'umap::umap': 'UMAP dimensionality reduction',
    'igraph::': 'graph network analysis',
    'sna::': 'social network analysis',
    'ergm::': 'exponential random graph model',
    'sp::': 'spatial data analysis',
    'sf::': 'simple features spatial data',
    'raster::': 'raster geospatial data',
    'terra::': 'geospatial raster vector data',

    # ── Julia language ────────────────────────────────────────────────
    'using Flux': 'Flux deep learning Julia',
    'Flux.Dense': 'dense layer Julia',
    'Flux.Conv': 'convolution layer Julia',
    'Flux.Chain': 'model chain Julia',
    'Flux.train!': 'training Julia',
    'using DifferentialEquations': 'differential equations Julia',
    'ODEProblem': 'ODE problem Julia',
    'SDEProblem': 'stochastic differential equation Julia',
    'solve(prob': 'solve differential equation Julia',
    'using Turing': 'Turing probabilistic programming Julia',
    'using Distributions': 'probability distributions Julia',
    'using Optim': 'optimization Julia',
    'using JuMP': 'mathematical optimization JuMP Julia',
    'using LinearAlgebra': 'linear algebra Julia',
    'using Statistics': 'statistics Julia',
    'using DataFrames': 'data frames Julia',
    'using MLJ': 'machine learning Julia',
    'using Graphs': 'graph analysis Julia',
    'using Zygote': 'automatic differentiation Julia',
    'Zygote.gradient': 'gradient automatic differentiation',

    # ── MATLAB patterns ───────────────────────────────────────────────
    'zeros(': 'zero matrix initialization',
    'ones(': 'ones matrix initialization',
    'eye(': 'identity matrix',
    'rand(': 'random uniform matrix',
    'randn(': 'random normal matrix',
    'linspace(': 'linearly spaced vector',
    'meshgrid(': 'coordinate grid mesh',
    'fft(': 'fast Fourier transform FFT',
    'ifft(': 'inverse FFT',
    'fft2(': '2D Fourier transform',
    'conv2(': '2D convolution',
    'imfilter(': 'image filtering',
    'eig(': 'eigenvalue decomposition',
    'svd(': 'singular value decomposition SVD',
    'pinv(': 'pseudoinverse Moore-Penrose',
    'fminunc(': 'unconstrained optimization',
    'fmincon(': 'constrained optimization',
    'lsqnonlin(': 'nonlinear least squares',
    'ode45(': 'ODE Runge-Kutta solver',
    'ode23(': 'ODE solver low order',
    'pdepe(': 'PDE solver',
    'fitlm(': 'linear model fitting',
    'fitglm(': 'generalized linear model fitting',
    'ClassificationTree': 'classification decision tree',
    'TreeBagger': 'random forest bagged trees',
    'fitcsvm(': 'SVM classification',
    'fitcecoc(': 'multiclass SVM',
    'kmeans(': 'k-means clustering',
    'linkage(': 'hierarchical clustering linkage',
    'pca(': 'principal component analysis',
    'tsne(': 't-SNE visualization',
    'crossval(': 'cross-validation',
    'perfcurve(': 'ROC performance curve',
    'confusionmat(': 'confusion matrix',
    'trainNetwork(': 'deep learning training MATLAB',
    'dlnetwork(': 'deep learning network MATLAB',

    # ── Pandas data manipulation ──────────────────────────────────────
    'pd.DataFrame': 'data frame creation',
    'pd.read_csv': 'CSV data loading',
    'pd.merge(': 'data merge join',
    '.groupby(': 'data grouping aggregation',
    '.pivot_table(': 'pivot table aggregation',
    '.fillna(': 'missing value imputation',
    '.dropna(': 'missing value removal',
    '.rolling(': 'rolling window computation',
    '.resample(': 'time series resampling',
    '.apply(': 'apply function',
    '.agg(': 'aggregation',
    '.value_counts(': 'frequency counting',
    '.corr(': 'correlation matrix',
    '.describe(': 'descriptive statistics summary',

    # ── PyTorch utilities ─────────────────────────────────────────────
    'torch.arange': 'index generation range',
    'torch.meshgrid': 'coordinate grid',
    'torch.linspace': 'linearly spaced tensor',
    'torch.zeros(': 'zero tensor initialization',
    'torch.ones(': 'ones tensor initialization',
    'torch.randn(': 'random normal tensor',
    'torch.rand(': 'random uniform tensor',
    'torch.eye(': 'identity matrix tensor',
    'torch.clamp(': 'value clamping clipping',
    'torch.where(': 'conditional selection',
    'torch.gather(': 'gather indexing',
    'torch.scatter(': 'scatter indexing',
    'torch.no_grad': 'gradient disabled inference',
    'torch.cuda.amp': 'automatic mixed precision training',
    'autocast': 'automatic mixed precision',
    'GradScaler': 'gradient scaling mixed precision',
    'DataLoader': 'data loading batching',
    'Dataset': 'dataset class',
    'DistributedDataParallel': 'distributed training data parallel',
    'DataParallel': 'data parallelism multi-GPU',
    'torch.save': 'model saving checkpoint',
    'torch.load': 'model loading checkpoint',
    'torch.jit': 'JIT compilation tracing scripting',
    'torch.onnx': 'ONNX export model conversion',

    # ── Image / audio / signal processing ─────────────────────────────
    'PIL.Image': 'image loading processing',
    'skimage': 'scikit-image image processing',
    'skimage.filters': 'image filtering',
    'skimage.segmentation': 'image segmentation',
    'skimage.feature': 'image feature extraction',
    'skimage.morphology': 'morphological operations image',
    'skimage.transform': 'image geometric transformation',
    'librosa': 'audio analysis processing',
    'librosa.stft': 'short-time Fourier transform audio',
    'librosa.feature.mfcc': 'MFCC mel-frequency cepstral coefficients',
    'librosa.feature.melspectrogram': 'mel spectrogram audio',
    'torchaudio': 'audio processing PyTorch',
    'soundfile': 'audio file reading writing',

    # ── Bioinformatics ────────────────────────────────────────────────
    'Bio.Seq': 'biological sequence DNA RNA protein',
    'Bio.PDB': 'protein structure PDB',
    'Bio.Align': 'sequence alignment bioinformatics',
    'scanpy': 'single-cell RNA-seq analysis',
    'anndata': 'annotated data single-cell',
    'DESeq2': 'differential expression RNA-seq',
    'edgeR': 'differential expression RNA-seq',
    'limma': 'linear models microarray differential expression',
    'Seurat': 'single-cell RNA-seq analysis R',
    'phyloseq': 'microbiome phylogenetic analysis',
    'Biostrings': 'biological sequences R Bioconductor',
    'GenomicRanges': 'genomic ranges intervals',
    'BSgenome': 'genome sequences Bioconductor',
    'VariantAnnotation': 'variant annotation VCF',
    'clusterProfiler': 'gene set enrichment analysis',
    'BLAST': 'sequence alignment homology',
    'muscle': 'multiple sequence alignment',
    'hmmer': 'hidden Markov model profile search',
    'AlphaFold': 'protein structure prediction',
    'ESM': 'protein language model',
    'RDKit': 'molecular chemistry cheminformatics',
    'openbabel': 'molecular conversion chemistry',
    'MDAnalysis': 'molecular dynamics simulation analysis',

    # ── Causal inference / econometrics ───────────────────────────────
    'DoWhy': 'causal inference framework',
    'CausalImpact': 'causal impact Bayesian structural time series',
    'instrumental_variable': 'instrumental variable regression',
    'propensity_score': 'propensity score matching',
    'diff_in_diff': 'difference-in-differences causal',
    'regression_discontinuity': 'regression discontinuity design',
    'ATE': 'average treatment effect causal',
    'CATE': 'conditional average treatment effect',
    'uplift': 'uplift modeling treatment effect',
    'fixest::feols': 'fixed effects regression',
    'plm::plm': 'panel data linear model',
    'ivreg(': 'instrumental variable regression',
    'MatchIt': 'matching causal inference',
    'twang': 'propensity score weighting',
    'AER::ivreg': 'instrumental variable two-stage least squares',
    'rdrobust': 'regression discontinuity robust',
    'synthdid': 'synthetic difference-in-differences',
    'gsynth': 'generalized synthetic control',
    'did::att_gt': 'difference-in-differences group time',

    # ── Optimal transport / Wasserstein ───────────────────────────────
    'ot.emd(': 'earth mover distance optimal transport',
    'ot.sinkhorn': 'Sinkhorn optimal transport entropic regularization',
    'wasserstein_distance': 'Wasserstein distance optimal transport',
    'geomloss': 'geometric loss Sinkhorn',
    'ot.bregman': 'Bregman projection optimal transport',

    # ── Symbolic / automatic differentiation ──────────────────────────
    'sympy': 'symbolic computation mathematics',
    'autograd': 'automatic differentiation',
    '.backward(': 'backpropagation gradient computation',
    '.grad': 'gradient',
    'torch.autograd': 'automatic differentiation PyTorch',
    'tf.GradientTape': 'automatic differentiation gradient computation',
    'jax.grad': 'automatic differentiation gradient JAX',
    'jax.jacobian': 'Jacobian matrix',
    'jax.hessian': 'Hessian matrix second-order derivative',
    'torch.autograd.functional.jacobian': 'Jacobian matrix',
    'torch.autograd.functional.hessian': 'Hessian matrix',

    # ── Geometric deep learning / 3D ──────────────────────────────────
    'PointNet': 'PointNet point cloud',
    'point_cloud': 'point cloud 3D',
    'mesh': 'mesh 3D geometry',
    'voxel': 'voxel 3D grid',
    'kaolin': 'Kaolin 3D deep learning',
    'open3d': 'Open3D point cloud 3D',
    'trimesh': 'triangle mesh 3D',
    'pytorch3d': 'PyTorch3D 3D vision',
    'NeRF': 'neural radiance field NeRF',
    'sdf': 'signed distance function SDF',
    'occupancy': 'occupancy network 3D',
    'ray_marching': 'ray marching rendering',
    'volume_rendering': 'volume rendering neural',
}


# Boilerplate prefixes to skip in doc-comments (language-agnostic)
_DOC_BOILERPLATE = (
    'Args:', 'Returns:', 'Raises:', 'Example:', 'Note:', 'Yields:',
    'Attributes:', 'Parameters:', 'See Also:', 'References:', 'Todo:',
    '@param', '@return', '@returns', '@throws', '@exception', '@see',
    '@since', '@deprecated', '@author', '@version', '@type', '@brief',
    '\\param', '\\return', '\\brief', '\\note',  # Doxygen
    ':param', ':returns:', ':rtype:', ':raises:',  # Sphinx
)

# Comment-line noise to filter out
_COMMENT_NOISE = ('!', '-*-', 'type:', 'noqa', 'nolint', 'pylint', 'noinspection',
                  'eslint', 'pragma', 'TODO', 'FIXME', 'HACK', 'XXX', 'NOLINT',
                  'skipcq', 'rubocop', '@ts-', 'istanbul', 'c8 ignore')

# Trivial identifiers to skip across all languages
_TRIVIAL_NAMES = frozenset({
    'init', 'main', 'self', 'cls', 'forward', 'call', 'new', 'run',
    'get', 'set', 'test', 'setup', 'teardown', 'toString', 'equals',
    'constructor', 'destructor', 'dispose', 'finalize',
})


def _extract_doc_comments(code: str) -> list[str]:
    """Extract documentation strings/comments from code in any language.

    Supports Python triple-quoted strings, JSDoc/Javadoc/Doxygen block comments,
    R roxygen (#'), Julia docstrings, and MATLAB %{ %} blocks.
    """
    docs: list[str] = []

    # Python / Julia triple-quoted docstrings
    for m in re.finditer(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', code, re.DOTALL):
        doc = (m.group(1) or m.group(2)).strip()
        if doc:
            docs.append(doc)

    # JSDoc / Javadoc / Doxygen: /** ... */ or /*! ... */
    for m in re.finditer(r'/\*[*!](.*?)\*/', code, re.DOTALL):
        doc = m.group(1).strip()
        # Strip leading * on each line
        lines = [re.sub(r'^\s*\*\s?', '', line) for line in doc.split('\n')]
        doc = '\n'.join(lines).strip()
        if doc:
            docs.append(doc)

    # R roxygen comments: #' lines
    roxygen_lines: list[str] = []
    for m in re.finditer(r"#'\s*(.+)", code):
        roxygen_lines.append(m.group(1).strip())
    if roxygen_lines:
        docs.append('\n'.join(roxygen_lines))

    # MATLAB block comments: %{ ... %}
    for m in re.finditer(r'%\{(.*?)%\}', code, re.DOTALL):
        doc = m.group(1).strip()
        if doc:
            docs.append(doc)

    # MATLAB inline doc: consecutive % comment lines at function start
    matlab_doc: list[str] = []
    for m in re.finditer(r'%\s*(.+)', code):
        line = m.group(1).strip()
        if line and not line.startswith('{') and not line.startswith('}'):
            matlab_doc.append(line)
    # Only use if we didn't already get block comments
    if matlab_doc and not any('%{' in d for d in docs):
        # Take only consecutive comment lines (first block)
        pass  # These get picked up by the comment extractor below

    return docs


def _extract_comments(code: str) -> list[str]:
    """Extract single-line comments from code in any language.

    Supports: # (Python/R/Julia/Shell), // (C/C++/Java/JS/Go/Rust),
    % (MATLAB/LaTeX), -- (Lua/SQL/Haskell).
    """
    comments: list[str] = []

    # # comments (Python, R, Julia, Ruby, Shell, Perl) — but not #include, #' (roxygen), #! (shebang)
    for m in re.finditer(r'(?<!&)#(?![\'!{])\s*(.+)', code):
        comments.append(m.group(1).strip())

    # // comments (C, C++, Java, JavaScript, TypeScript, Go, Rust, C#, Swift, Kotlin)
    for m in re.finditer(r'//\s*(.+)', code):
        comment = m.group(1).strip()
        if not comment.startswith('/'):  # Skip /// which is doc-comment (already captured)
            comments.append(comment)

    # % comments (MATLAB, LaTeX, Octave) — but not %{ or %}
    for m in re.finditer(r'(?<![\\])%(?![{}\d])\s*(.+)', code):
        comments.append(m.group(1).strip())

    # -- comments (Lua, SQL, Haskell)
    for m in re.finditer(r'--\s*(.+)', code):
        comments.append(m.group(1).strip())

    # Filter: ≥3 words, not noise
    filtered: list[str] = []
    for c in comments:
        if len(c.split()) >= 3 and not any(c.startswith(n) for n in _COMMENT_NOISE):
            filtered.append(c)

    return filtered


def _extract_function_names(code: str) -> list[str]:
    """Extract function/class/method names from code in any language.

    Supports: def/class (Python), function (JS/R/MATLAB/Lua/Julia),
    fn (Rust), func (Go/Swift), fun (Kotlin), sub/method (Perl),
    public/private/protected methods (Java/C#/C++),
    struct/enum/trait/impl (Rust), interface (Go/Java/TS).
    """
    names: list[str] = []

    patterns = [
        # Python: def/class
        r'(?:def|class)\s+(\w+)',
        # JavaScript/TypeScript/R/MATLAB/Lua/PHP: function name
        r'\bfunction\s+(\w+)',
        # JS/TS: const/let/var name = (arrow functions)
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])\s*=>',
        # Rust: fn/struct/enum/trait/impl
        r'\b(?:fn|struct|enum|trait|impl)\s+(\w+)',
        # Go: func/type
        r'\b(?:func|type)\s+(\w+)',
        # Swift/Kotlin: func/fun/class/struct/enum/protocol
        r'\b(?:func|fun)\s+(\w+)',
        # Java/C#/C++: class/interface + access modifier methods
        r'\b(?:class|interface|struct|enum)\s+(\w+)',
        r'\b(?:public|private|protected|static)\s+(?:\w+\s+)*(\w+)\s*\(',
        # Julia: function name / struct
        r'\bfunction\s+(\w+)',
        r'\b(?:mutable\s+)?struct\s+(\w+)',
        # R: name <- function
        r'(\w+)\s*<-\s*function\s*\(',
        # R: name = function
        r'(\w+)\s*=\s*function\s*\(',
        # MATLAB: function [out] = name(in)
        r'\bfunction\s+(?:\[?\w+(?:,\s*\w+)*\]?\s*=\s*)?(\w+)\s*\(',
        # Perl: sub name
        r'\bsub\s+(\w+)',
        # Lua: function name or local function name
        r'\blocal\s+function\s+(\w+)',
    ]

    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, code):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            # Strip leading underscores
            clean = name.lstrip('_') if name.startswith('_') and not name.startswith('__') else name
            # Split camelCase and snake_case into words
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
            words = words.replace('_', ' ').replace('.', ' ').strip()
            if words and words.lower() not in _TRIVIAL_NAMES:
                names.append(words)

    return names


def _get_first_doc_sentence(code: str) -> str | None:
    """Get the first meaningful sentence from any doc-comment in the code."""
    docs = _extract_doc_comments(code)
    for doc in docs:
        for line in doc.split('\n'):
            line = line.strip()
            if line and not any(line.startswith(bp) for bp in _DOC_BOILERPLATE) and len(line.split()) >= 3:
                return line.rstrip('.')
    return None


def _code_to_search_query(code: str) -> str:
    """Transform code into a natural-language query for paper search.

    Language-agnostic: extracts doc-comments, inline comments, identifier
    names, and detects algorithmic operations to build a query that maps
    code patterns to paper prose.
    """
    parts: list[str] = []

    # 1. Detect algorithmic operations from code patterns
    operations_found: list[str] = []
    code_lower = code.lower()
    for pattern, description in CODE_OPERATION_MAP.items():
        if pattern.lower() in code_lower:
            operations_found.append(description)
    if operations_found:
        seen = set()
        for op in operations_found:
            if op not in seen:
                seen.add(op)
                parts.append(op)

    # 2. Extract doc-comments — first sentence only
    first_sent = _get_first_doc_sentence(code)
    if first_sent:
        parts.append(first_sent)

    # 3. Extract inline comments (≥3 words, language-agnostic)
    comments = _extract_comments(code)
    parts.extend(comments)

    # 4. Extract function/class names (language-agnostic)
    names = _extract_function_names(code)
    parts.extend(names)

    if parts:
        return " ".join(parts)
    return code


def _extract_matchable_phrases(code: str) -> list[str]:
    """Extract phrases from code that might appear verbatim in a paper.

    Language-agnostic: pulls out doc-comment sentences, inline comments,
    and function/class names as natural language phrases.
    """
    phrases: list[str] = []

    # 1. Doc-comment sentences (skip boilerplate lines)
    for doc in _extract_doc_comments(code):
        for line in doc.split('\n'):
            line = line.strip()
            if (line
                    and not any(line.startswith(bp) for bp in _DOC_BOILERPLATE)
                    and len(line.split()) >= 3):
                phrases.append(line.rstrip('.'))

    # 2. Inline comments (≥3 words, language-agnostic)
    phrases.extend(_extract_comments(code))

    # 3. Function/class names as natural language
    phrases.extend(_extract_function_names(code))

    return phrases


def _direct_text_search(
    phrases: list[str],
    paper_collection,
    max_results_per_phrase: int = 3,
) -> list[dict]:
    """Search paper chunks for near-verbatim matches using ChromaDB $contains.

    This guarantees that phrases appearing in both code and paper are surfaced
    regardless of embedding quality.
    """
    seen_ids: set[str] = set()
    results: list[dict] = []

    for phrase in phrases:
        # Skip very short phrases that would match too broadly
        if len(phrase) < 8:
            continue
        try:
            matches = paper_collection.get(
                where_document={"$contains": phrase.lower()},
                include=["documents", "metadatas"],
            )
            if matches and matches["ids"]:
                for i, doc_id in enumerate(matches["ids"][:max_results_per_phrase]):
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        results.append({
                            "id": doc_id,
                            "document": matches["documents"][i],
                            "metadata": matches["metadatas"][i],
                            "similarity": 0.95,  # High score for direct text match
                        })
        except Exception:
            # ChromaDB $contains may fail on some inputs; skip gracefully
            continue

    return results


def _parse_index(raw) -> int | None:
    """Parse an index value that may be an int, string int, or prefixed like 'snippet_2'."""
    if raw is None:
        return None
    s = str(raw).strip()
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else None


async def analyze_highlight(
    highlighted_text: str,
    codebase: Optional[Codebase] = None,
    n_results: int = 5,
) -> HighlightAnalysisResponse:
    from backend.services.precompute import get_precompute_cache

    from backend.services.logging import logger

    cache = get_precompute_cache()
    cached = cache.get(highlighted_text)
    if cached is not None:
        logger.info(f"[highlight] cache HIT — returning precomputed result")
        return cached
    logger.info(f"[highlight] cache MISS — running live pipeline")

    if codebase is None:
        codebase = app_state.codebase

    if codebase is None:
        raise ValueError("No codebase loaded")

    settings = get_settings()
    embedding_service = get_embedding_service()

    # Get more results for hybrid search and reranking
    rerank_k = settings.rerank_top_k if settings.use_reranking else n_results
    embedding_results = await embedding_service.search_similar_async(
        highlighted_text, n_results=rerank_k
    )

    from backend.services.logging import logger
    logger.debug(f"[highlight] embedding_results: {len(embedding_results)}, rerank_k: {rerank_k}")

    if not embedding_results:
        return HighlightAnalysisResponse(
            highlighted_text=highlighted_text,
            code_references=[],
            summary="No relevant code found for the highlighted text.",
        )

    # Apply hybrid search (combine embedding with BM25)
    hybrid_searcher = get_hybrid_searcher()
    similar_chunks = await hybrid_searcher.search_code_hybrid(
        query=highlighted_text,
        embedding_results=embedding_results,
        n_results=rerank_k,
    )

    logger.debug(f"[highlight] after hybrid: {len(similar_chunks)}")

    # Apply reranking if enabled
    if settings.use_reranking and similar_chunks:
        reranker = get_reranker()
        similar_chunks = reranker.rerank_for_code(
            highlighted_text=highlighted_text,
            code_results=similar_chunks,
            top_k=n_results,
        )
    else:
        similar_chunks = similar_chunks[:n_results]

    logger.debug(f"[highlight] after rerank: {len(similar_chunks)}")

    if not similar_chunks:
        return HighlightAnalysisResponse(
            highlighted_text=highlighted_text,
            code_references=[],
            summary="No relevant code found for the highlighted text.",
        )

    chunks_text = "\n\n".join(
        f"[Snippet {i} — {chunk['metadata']['relative_path']}:{chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}]\n{chunk['document']}"
        for i, chunk in enumerate(similar_chunks)
    )

    prompt = f"""Highlighted text from paper:
\"\"\"{highlighted_text}\"\"\"

Potentially relevant code snippets:
{chunks_text}

Analyze which snippets are most relevant to the highlighted text."""

    ai_provider = get_ai_provider()
    response = await ai_provider.complete(
        prompt,
        system_prompt=HIGHLIGHT_SYSTEM_PROMPT,
        temperature=0.0,
        response_format=HIGHLIGHT_RESPONSE_SCHEMA,
    )
    logger.debug(f"[highlight] LLM response (first 500): {response[:500]}")

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

    logger.debug(f"[highlight] parsed keys: {list(result.keys())}, snippets: {result.get('relevant_snippets', 'MISSING')}")

    code_references = []
    for snippet_info in result.get("relevant_snippets", []):
        if not isinstance(snippet_info, dict):
            continue
        raw_idx = snippet_info.get("index", snippet_info.get("snippet_id", snippet_info.get("id")))
        idx = _parse_index(raw_idx)
        if idx is None:
            continue
        if idx < len(similar_chunks):
            chunk = similar_chunks[idx]
            metadata = chunk["metadata"]

            lines = chunk["document"].split("\n")
            content_start = next(
                (i for i, line in enumerate(lines) if not line.startswith("File:")),
                0,
            )
            content = "\n".join(lines[content_start:])

            explanation = (
                snippet_info.get("explanation")
                or snippet_info.get("reason")
                or snippet_info.get("description")
                or "Matched via semantic similarity"
            )
            relevance_score = min(1.0, max(0.0, float(snippet_info.get(
                "relevance_score", snippet_info.get("score", chunk["similarity"])
            ))))

            code_references.append(
                CodeReference(
                    file_path=metadata["file_path"],
                    relative_path=metadata["relative_path"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    content=content,
                    relevance_score=relevance_score,
                    explanation=explanation,
                )
            )

    logger.debug(f"[highlight] code_references from LLM: {len(code_references)}")

    # Fallback: if LLM returned no usable references, use raw search results
    if not code_references and similar_chunks:
        logger.debug(f"[highlight] fallback triggered, using {len(similar_chunks)} raw results")
        for chunk in similar_chunks[:n_results]:
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
                    relevance_score=min(1.0, max(0.0, float(chunk.get("similarity", 0.5)))),
                    explanation="Matched via semantic similarity",
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
    response = await ai_provider.complete(prompt, system_prompt=ALIGNMENT_SYSTEM_PROMPT, temperature=0.0)

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


async def analyze_code_highlight(
    highlighted_code: str,
    file_path: Optional[str] = None,
    paper: Optional[Paper] = None,
    n_results: int = 5,
) -> CodeHighlightAnalysisResponse:
    """Analyze highlighted code and find relevant paper sections."""
    if paper is None:
        paper = app_state.paper

    if paper is None:
        raise ValueError("No paper loaded")

    settings = get_settings()
    embedding_service = get_embedding_service()

    # --- Multi-query search with result fusion ---
    # Query 1: Algorithm-aware query (operation descriptions + docstring + names)
    search_query = _code_to_search_query(highlighted_code)

    # Query 2: Doc-comment-focused query (first sentence, language-agnostic)
    docstring_query = _get_first_doc_sentence(highlighted_code)

    # Query 3: Raw code (truncated) — catches exact identifier overlap
    raw_code_query = highlighted_code[:500] if len(highlighted_code) > 500 else highlighted_code

    rerank_k = settings.rerank_top_k if settings.use_reranking else n_results

    # Run all embedding searches
    all_results: dict[str, dict] = {}  # id -> result (deduped)
    hit_counts: dict[str, int] = {}    # id -> number of queries that found it

    queries = [search_query]
    if docstring_query and docstring_query != search_query:
        queries.append(docstring_query)
    if raw_code_query != search_query:
        queries.append(raw_code_query)

    for query in queries:
        results = await embedding_service.search_paper_similar(
            query, n_results=rerank_k
        )
        for r in results:
            rid = r["id"]
            if rid not in all_results:
                all_results[rid] = r
                hit_counts[rid] = 1
            else:
                hit_counts[rid] += 1
                # Keep the higher similarity score
                if r["similarity"] > all_results[rid]["similarity"]:
                    all_results[rid] = r

    # Boost scores for documents found by multiple queries
    for rid, count in hit_counts.items():
        if count > 1:
            boost = 1.0 + 0.1 * (count - 1)
            all_results[rid]["similarity"] = min(1.0, all_results[rid]["similarity"] * boost)

    # --- Direct text-match retrieval ---
    phrases = _extract_matchable_phrases(highlighted_code)
    if phrases:
        paper_collection = embedding_service._paper_collection
        if paper_collection is None:
            try:
                paper_collection = embedding_service.chroma_client.get_collection("paper_chunks")
            except Exception:
                paper_collection = None
        if paper_collection is not None:
            direct_matches = _direct_text_search(phrases, paper_collection)
            for r in direct_matches:
                rid = r["id"]
                if rid not in all_results:
                    all_results[rid] = r
                else:
                    # Direct match boosts existing result
                    all_results[rid]["similarity"] = min(1.0, max(
                        all_results[rid]["similarity"], r["similarity"]
                    ))

    # Sort by similarity and take top rerank_k
    embedding_results = sorted(all_results.values(), key=lambda x: x["similarity"], reverse=True)[:rerank_k]

    if not embedding_results:
        return CodeHighlightAnalysisResponse(
            highlighted_code=highlighted_code,
            paper_references=[],
            summary="No relevant paper sections found for the highlighted code.",
        )

    # Apply hybrid search — lower embedding weight for code→paper (BM25 matters more)
    hybrid_searcher = get_hybrid_searcher()
    similar_sections = await hybrid_searcher.search_paper_hybrid(
        query=search_query,
        embedding_results=embedding_results,
        n_results=rerank_k,
        embedding_weight=0.45,
    )

    # Apply reranking if enabled
    if settings.use_reranking and similar_sections:
        reranker = get_reranker()
        similar_sections = reranker.rerank_for_paper(
            code_snippet=highlighted_code,
            paper_results=similar_sections,
            top_k=n_results,
        )
    else:
        similar_sections = similar_sections[:n_results]

    if not similar_sections:
        return CodeHighlightAnalysisResponse(
            highlighted_code=highlighted_code,
            paper_references=[],
            summary="No relevant paper sections found for the highlighted code.",
        )

    # Format context for AI
    context_info = f"Code from file: {file_path}\n" if file_path else ""
    sections_text = "\n\n".join(
        f"[Section {i}] Page {section['metadata'].get('page', 'N/A')}:\n{section['document']}"
        for i, section in enumerate(similar_sections)
    )

    prompt = f"""Highlighted code:
{context_info}
\"\"\"{highlighted_code}\"\"\"

Potentially relevant paper sections:
{sections_text}

Analyze which paper sections are most relevant to this code implementation."""

    ai_provider = get_ai_provider()
    response = await ai_provider.complete(
        prompt,
        system_prompt=CODE_TO_PAPER_SYSTEM_PROMPT,
        temperature=0.0,
        response_format=CODE_HIGHLIGHT_RESPONSE_SCHEMA,
    )

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx]
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {
            "relevant_sections": [
                {"index": i, "relevance_score": section["similarity"], "explanation": ""}
                for i, section in enumerate(similar_sections[:3])
            ],
            "summary": "Analysis completed using embedding similarity.",
        }

    paper_references = []
    for section_info in result.get("relevant_sections", []):
        if not isinstance(section_info, dict):
            continue
        raw_idx = section_info.get("index", section_info.get("section_id", section_info.get("id")))
        idx = _parse_index(raw_idx)
        if idx is None:
            continue
        if idx < len(similar_sections):
            section = similar_sections[idx]
            metadata = section["metadata"]

            # Extract clean content (remove header lines)
            lines = section["document"].split("\n")
            content_start = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if not line.startswith("Paper:") and not line.startswith("Page ")
                ),
                0,
            )
            content = "\n".join(lines[content_start:])

            explanation = (
                section_info.get("explanation")
                or section_info.get("reason")
                or section_info.get("description")
                or "Matched via semantic similarity"
            )
            relevance_score = min(1.0, max(0.0, float(section_info.get(
                "relevance_score", section_info.get("score", section["similarity"])
            ))))

            paper_references.append(
                PaperReference(
                    content=content,
                    page=metadata.get("page"),
                    start_idx=metadata["start_idx"],
                    end_idx=metadata["end_idx"],
                    relevance_score=relevance_score,
                    explanation=explanation,
                )
            )

    # Fallback: if LLM returned no usable references, use raw search results
    if not paper_references and similar_sections:
        for section in similar_sections[:n_results]:
            metadata = section["metadata"]
            lines = section["document"].split("\n")
            content_start = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if not line.startswith("Paper:") and not line.startswith("Page ")
                ),
                0,
            )
            content = "\n".join(lines[content_start:])
            paper_references.append(
                PaperReference(
                    content=content,
                    page=metadata.get("page"),
                    start_idx=metadata["start_idx"],
                    end_idx=metadata["end_idx"],
                    relevance_score=min(1.0, max(0.0, float(section.get("similarity", 0.5)))),
                    explanation="Matched via semantic similarity",
                )
            )

    return CodeHighlightAnalysisResponse(
        highlighted_code=highlighted_code,
        paper_references=paper_references,
        summary=result.get("summary", ""),
    )
