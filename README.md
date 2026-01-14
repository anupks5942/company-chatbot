# CloudNexus Local RAG Bot

End-to-end, local-only retrieval-augmented chatbot for CloudNexus. It ingests PDFs and CSVs in `./data`, builds a FAISS index with Ollama embeddings, re-ranks hits with a lightweight keyword scorer, and answers questions with Llama 3.2 served by Ollama. Employee CSV rows are treated as authoritative ground truth.

## What You Get
- Local-only Llama 3.2 flow (no external APIs)
- Hybrid ingestion: PDFs (policies/info) plus CSV employee records
- Deterministic rerank: keyword-based boost, employee records prioritized
- Simple CLI loop with source listing for transparency

## Prerequisites
- Python 3.11 or 3.12 (recommended)
- Ollama running locally: install from https://ollama.com and start `ollama serve`
- Model pulled: `ollama pull llama3.2:3b`

## Quickstart (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-ollama faiss-cpu pypdf pandas

# Add your data before running
python main.py
```

## Project Layout
- [main.py](main.py): end-to-end pipeline (ingest, split, embed, index, retrieve, rerank, generate)
- [data/](data/): place PDFs and CSVs here (created manually)
- vectorstore/db_faiss: written at runtime after indexing

## Prepare Your Data
- PDFs: any company docs or policies. Place under `./data`.
- CSVs: employee records; expected columns (case-sensitive):
  - `employee_name`, `designation`, `team_lead`, `team_name`, `department`
  - Missing columns are handled as `Unknown`, but matching works best when present.

## Run the Bot
1) Ensure Ollama is running: `ollama serve`
2) Activate your venv: `venv\Scripts\activate`
3) Start the app: `python main.py`
4) Type questions; enter `exit` to quit.

Examples:
- Who leads the cloud security team?
- What department does Priya Sharma belong to?
- Summarize key points from the remote work policy.

## How It Works
1) Ingestion: loads all PDFs and CSVs from `./data`. CSV rows are converted into rich text snippets marked `[SOURCE: EMPLOYEE_DB]` and tagged as high-priority.
2) Splitting: `RecursiveCharacterTextSplitter` with chunk size 800 and overlap 100 keeps employee records intact while allowing reasonable policy chunks.
3) Embedding + Index: `OllamaEmbeddings` (model `llama3.2:3b`) feed `FAISS.from_documents`; index is saved to `vectorstore/db_faiss` each run.
4) Retrieval: FAISS retriever pulls top 10 by vector similarity.
5) Rerank: keyword scorer boosts records containing query terms; employee-tagged content is favored; top 5 are kept for the prompt.
6) Generation: `OllamaLLM` answers with a concise response following rules that prioritize the employee database and avoid fabricating people or salaries.
7) Transparency: the CLI prints which source files contributed to the answer.

## Configuration (edit in [main.py](main.py))
- `MODEL_NAME`: Ollama model name (default `llama3.2:3b`)
- `DATA_PATH`: folder to scan for PDFs/CSVs (default `./data`)
- `DB_FAISS_PATH`: output folder for the FAISS index
- Chunking: `chunk_size=800`, `chunk_overlap=100`
- Retrieval: `k=10`, rerank keeps top 5

## Operational Notes
- The index is rebuilt on every run (current script does not load an existing index).
- If `./data` is empty, the app exits early with a message.
- CSV parsing errors are printed but do not stop the run.

## Troubleshooting
- Ollama not running: start `ollama serve`; ensure the model is pulled.
- Missing deps: rerun pip install command above inside the venv.
- Empty answers: confirm there are PDFs/CSVs in `./data` and that query terms exist in the content.
- Slow first run: indexing and embedding can take time on CPU/GTX 1650; subsequent runs reuse saved vectors but still rebuild today.

## Extending
- Swap model: change `MODEL_NAME` to another Ollama model.
- Tune recall/precision: adjust `k` in the retriever or change rerank cutoff.
- Add loaders: extend ingestion to Word/HTML as needed.
