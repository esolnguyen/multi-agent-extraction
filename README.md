# Multi-Agent Document Extraction

Multi-agent parallel document extraction using **Gemini** for LLM and **Azure Document Intelligence** for OCR. Runs locally in Docker.

## How It Works

```
Input PDF
    |
    v
[OCR] Azure Document Intelligence
    |  -> words with IDs, bounding boxes, confidence
    v
[DATA_EXTRACTION] Multi-agent parallel (Gemini)
    |
    |-- Agent: header_info  -> title, date, doc number
    |-- Agent: line_items   -> descriptions, quantities, prices
    |-- Agent: totals       -> subtotal, tax, grand total
    |
    v
Merged final_result.json
```

Each agent runs in parallel, receives the same PDF + OCR data, but has its own prompt and response schema. Results are merged into a single output.

## Quick Start

```bash
# 1. Set up credentials
cp docker/.env.example docker/.env
# Edit docker/.env with your GEMINI_API_KEY, AZURE_DI_ENDPOINT, AZURE_DI_KEY

# 2. Add a PDF
cp /path/to/your/document.pdf data/input.pdf

# 3. Run
docker compose up --build
```

## Output

```
data/output/
  ocr_results.json      # OCR word IDs + bounding boxes
  final_result.json     # Merged extraction from all agents
  metrics.json          # Per-agent timing and token usage
  agents/
    header_info/result.json
    line_items/result.json
    totals/result.json
```

## Configuration

Edit `docker/flow_config.json` to customize:

- **Flow type**: `BASIC` (OCR + extraction) or `FILE_ONLY` (direct LLM, no OCR)
- **Agents**: add/remove agents, each with its own prompt and response schema
- **Model**: change `model_name` per agent or globally
- **OCR**: change the Azure DI model (e.g. `prebuilt-layout` for tables)

## Project Structure

```
src/
  run_flow.py          # Orchestrator - runs the full pipeline
  providers/
    gemini_llm.py      # Gemini API client
    azure_ocr.py       # Azure Document Intelligence client
  shared_llm/
    execution.py       # Core extract() function
    prompts.py         # System prompts for OCR and FILE_ONLY paths
    utils.py           # Word ID to bounding box mapping
  models/
    flow.py            # FlowStep, AgentExecution, LLMFlowStepConfig
    types.py           # LLMConfigOption
docker/
  flow_config.json     # Flow definition with agent prompts
  .env.example         # Environment variables template
  requirements.txt     # Python dependencies
```

## Run Without Docker

```bash
pip install google-genai azure-ai-documentintelligence pydantic
cd src
GEMINI_API_KEY=... AZURE_DI_ENDPOINT=... AZURE_DI_KEY=... python run_flow.py
```
