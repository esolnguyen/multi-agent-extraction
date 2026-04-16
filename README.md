# Multi-Agent Document Extraction

Multi-agent parallel document extraction using **Gemini** (LLM) and **Azure Document Intelligence** (OCR). Designed for large PDFs (hundreds of pages) where only a few pages carry the structured data you actually want.

## Pipeline

```
Input PDF (e.g. 300-page Owner's Handbook)
    |
    v
[1. PAGE_DETECTION]  Gemini scans the PDF in 100-page chunks (in parallel)
    |                 -> {engines: [45,46], tyres: [52], ...}
    v
[2. OCR]             Azure DI runs only on the union of detected pages
    |                 -> words with IDs, bounding boxes, confidence
    v
[3. DATA_EXTRACTION] Multi-agent parallel (Gemini)
    |                 Each agent gets: sliced PDF + filtered OCR for its pages
    |-- engines          -> engine/oil/fuel specs
    |-- tyres            -> sizes, pressures, wheel dims
    |-- transmissions    -> gearbox ratios, fluids
    |-- brakes           -> fluid standard, disc/pad specs
    |-- steering         -> fluid, capacities
    |-- axles            -> diff capacities, viscosity
    |-- service_interval -> maintenance tables
    v
Merged final_result.json
```

Why chunking in step 1: a 300-page PDF blows past context limits and burns tokens. Slicing into 100-page chunks lets Gemini reason about each chunk independently; results are remapped to absolute page numbers and merged.

Why filtering in step 3: each agent only sees the pages relevant to its domain — both the sliced PDF and the filtered OCR. No agent wastes tokens on the other 290 pages.

## Quick Start

```bash
# 1. Install deps (Python 3.11+)
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# edit .env: GEMINI_API_KEY, AZURE_DI_ENDPOINT, AZURE_DI_KEY

# 3. Drop a PDF in
cp /path/to/handbook.pdf data/input.pdf

# 4. Run
./run.sh
```

`run.sh` sources `.env` and runs `python src/run_flow.py`. Override inputs via env vars:

```bash
INPUT_PDF_PATH=/abs/path.pdf OUTPUT_DIR=/tmp/out ./run.sh
```

## Output

```
data/output/
  page_detection.json     # {domain: [page_numbers]}
  ocr_results.json        # OCR words with IDs + bboxes (only detected pages)
  final_result.json       # Merged extraction, keyed by domain
  metrics.json            # Per-agent timing + token usage
  agents/
    engines/result.json
    tyres/result.json
    ...
```

## Configuration

`flow_config.json` defines the pipeline. Key knobs:

- **Flow steps**: `PAGE_DETECTION`, `OCR`, `DATA_EXTRACTION` (ordered by `step_order`).
- **`pages_per_chunk`** (PAGE_DETECTION): default `100`.
- **Per-step `model_name`**: Gemini model (e.g. `gemini-2.5-flash`) or Azure DI model (e.g. `prebuilt-read`, `prebuilt-layout`).
- **Agents (`executions`)**: each has a `domain_key` (must match a key in `PAGE_DETECTION`'s `response_schema`), a `user_prompt`, and a `response_schema`.

To target a different document type: rewrite the PAGE_DETECTION prompt + its `response_schema` keys, then add matching agents under DATA_EXTRACTION.

## Project Structure

```
flow_config.json        # pipeline definition (prompts, schemas, models)
.env                    # credentials (gitignored)
run.sh                  # entrypoint: loads .env, runs pipeline
requirements.txt
pyproject.toml
src/
  run_flow.py           # orchestrator
  providers/
    gemini_llm.py       # Gemini client
    azure_ocr.py        # Azure DI client (supports page-range restriction)
  shared_llm/
    execution.py        # extract() — single LLM call with OCR TOON
    prompts.py          # system prompts (OCR path + FILE_ONLY path)
    utils.py            # wid -> bbox mapping
  models/
    flow.py, types.py   # pydantic models
data/
  input.pdf             # your PDF
  output/               # results
```

## Environment

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Gemini API |
| `GEMINI_MODEL` | no | Fallback model when `flow_config.json` agents don't specify `model_name` |
| `AZURE_DI_ENDPOINT` | yes | Azure Document Intelligence endpoint |
| `AZURE_DI_KEY` | yes | Azure DI key |
| `FLOW_CONFIG_PATH` | no | Override path to `flow_config.json` |
| `INPUT_PDF_PATH` | no | Override path to input PDF |
| `OUTPUT_DIR` | no | Override output directory |
