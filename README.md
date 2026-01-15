# Valid8

Research Paper & Code Alignment Checker - An AI-powered tool to verify alignment between research papers and their associated codebases.

## Features

- **Paper-to-Code Mapping**: Upload a research paper (PDF or text), highlight sections, and see related code with line numbers
- **Alignment Checking**: Write your own summary of what code should do, and check if the implementation matches
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

### 3. Start the Backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Open the Frontend

Open `frontend/index.html` in your browser, or serve it with:

```bash
python3 -m http.server 3000 --directory frontend
```

Then visit `http://localhost:3000`.

## Usage

### Highlight-to-Code Mode

1. Upload a research paper (PDF, TXT, MD, or TEX) or paste text directly
2. Load a codebase by entering a local path or GitHub URL
3. Wait for indexing to complete
4. Select/highlight text in the paper
5. View related code references with relevance scores

### Alignment Check Mode

1. Load a codebase
2. Switch to "Alignment Check" mode
3. Write a summary describing what the code should do
4. Click "Check Alignment" to see:
   - Alignment score (0-100%)
   - Issues found (missing, incorrect, or extra functionality)
   - Suggestions for improvement

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/papers/upload` | POST | Upload a paper file |
| `/api/papers/text` | POST | Submit paper as text |
| `/api/code/load` | POST | Load codebase (path or GitHub URL) |
| `/api/analysis/highlight` | POST | Analyze highlighted text |
| `/api/analysis/alignment` | POST | Check summary alignment |
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
