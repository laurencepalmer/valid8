# Valid8

Research Paper & Code Alignment Checker - An AI-powered tool to verify alignment between research papers and their associated codebases.

## Features

- **Paper-to-Code Mapping**: Upload a research paper (PDF or text), highlight sections, and see related code with line numbers
- **Code-to-Paper Mapping**: Select code in the browser to find related paper sections
- **Audit Mode**: Automatically verify code against paper claims, detect data leakage, evaluation errors, and other critical issues
- **Multiple AI Providers**: Support for OpenAI, Anthropic, or local Ollama models
- **Hybrid Embeddings**: Uses local sentence-transformers by default with API fallback

## Setup

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AI Provider

Copy the example environment file and configure your preferred AI provider:

```bash
cp .env.example .env
```

Edit `.env` to set your preferred provider:
- **Ollama (default)**: No API key needed, just have Ollama running locally
- **OpenAI**: Set `AI_PROVIDER=openai` and add your `OPENAI_API_KEY`
- **Anthropic**: Set `AI_PROVIDER=anthropic` and add your `ANTHROPIC_API_KEY`

### 3. Start the Server

```bash
uvicorn backend.main:app --reload
```

Then visit `http://localhost:8000`.

## Usage

### Paper-to-Code Search

1. Upload a research paper (PDF, TXT, MD, or TEX) or paste text directly
2. Load a codebase by entering a local path or GitHub URL
3. Wait for indexing to complete
4. Select/highlight text in the paper
5. View related code references with relevance scores

### Code-to-Paper Search

1. Load both a paper and codebase
2. Browse files in the code panel and select a file
3. Highlight code to find related paper sections
4. The paper will scroll to and highlight matching content

### Audit Mode

1. Load both a paper and codebase
2. Click "Run Audit" in the sidebar
3. The audit will automatically:
   - Extract claims from the paper
   - Analyze code behaviors
   - Detect catastrophic patterns (data leakage, evaluation errors, etc.)
   - Check alignment between claims and code
4. View the report with:
   - Overall alignment score
   - Critical warnings (tiered by severity)
   - Misalignments between paper and code
   - Verified and unverified claims

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/papers/upload` | POST | Upload a paper file |
| `/api/papers/text` | POST | Submit paper as text |
| `/api/code/load` | POST | Load codebase (path or GitHub URL) |
| `/api/analysis/highlight` | POST | Analyze highlighted paper text |
| `/api/analysis/code-highlight` | POST | Analyze highlighted code |
| `/api/audit/run` | POST | Start an audit |
| `/api/audit/{id}/status` | GET | Get audit progress |
| `/api/audit/{id}/report` | GET | Get audit report |
| `/api/status` | GET | Get current system status |

## Project Structure

```
valid8/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── api/routes/          # API endpoints
│   ├── services/            # Core services
│   │   ├── ai/              # AI provider adapters
│   │   ├── paper_parser.py  # PDF/text parsing
│   │   ├── code_loader.py   # Code loading
│   │   ├── embeddings.py    # Vector embeddings
│   │   └── alignment.py     # Analysis engine
│   └── models/              # Pydantic models
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/                  # Frontend JavaScript
└── requirements.txt
```

## Requirements

- Python 3.11+
- For local LLM: [Ollama](https://ollama.ai/) with a model like `llama3.2`
- For local embeddings: ~500MB disk space for the model download
