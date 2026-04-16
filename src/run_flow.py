import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).parent))

from shared_llm.execution import extract
from providers.azure_ocr import call_ocr
from providers.gemini_llm import call_gemini


def slice_pdf(pdf_bytes: bytes, pages: list[int]) -> bytes:
    """Return a new PDF containing only the given 1-based pages (in order)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    n = len(reader.pages)
    for p in sorted(set(pages)):
        if 1 <= p <= n:
            writer.add_page(reader.pages[p - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_flow")


# -- Page detection step --

def _detect_chunk(
    pd_config: dict,
    chunk_bytes: bytes,
    start_page: int,
    end_page: int,
) -> dict[str, list[int]]:
    """Run page detection on a single chunk. Returns domain -> absolute page numbers."""
    user_prompt = pd_config.get("user_prompt", "Return the page numbers per domain.")
    user_prompt = (
        f"{user_prompt}\n\n"
        f"NOTE: This PDF is a chunk of a larger document. Its pages correspond to "
        f"absolute pages {start_page}..{end_page} of the original document. "
        f"Return the LOCAL 1-based page numbers within this chunk "
        f"(1..{end_page - start_page + 1}); the caller will remap to absolute."
    )

    response = call_gemini(
        model_name=pd_config["model_name"],
        system_prompt=pd_config["system_prompt"],
        user_prompt=user_prompt,
        file_bytes=chunk_bytes,
        response_schema=pd_config.get("response_schema"),
    )
    parsed = json.loads(response["content"]) or {}

    result: dict[str, list[int]] = {}
    chunk_len = end_page - start_page + 1
    for domain, pages in parsed.items():
        if not isinstance(pages, list):
            continue
        abs_pages = []
        for p in pages:
            if isinstance(p, (int, float, str)) and str(p).lstrip("-").isdigit():
                local = int(p)
                if 1 <= local <= chunk_len:
                    abs_pages.append(start_page + local - 1)
        if abs_pages:
            result[domain] = sorted(set(abs_pages))
    return result


def run_page_detection_step(
    pdf_bytes: bytes,
    pd_config: dict,
    output_dir: Path,
) -> dict[str, list[int]]:
    """Ask the LLM which pages belong to which domain.

    Chunks the PDF into `pages_per_chunk` (default 100) page slices and calls
    the LLM in parallel per chunk, then remaps each chunk's local page numbers
    to absolute page numbers and merges.

    Returns: {domain_key: [absolute_page_numbers]} with empty domains omitted.
    """
    logger.info("=== STEP: PAGE_DETECTION ===")
    t0 = time.time()

    chunk_size = int(pd_config.get("pages_per_chunk", 100))
    total_pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    chunks = [(s, min(s + chunk_size - 1, total_pages)) for s in range(1, total_pages + 1, chunk_size)]
    logger.info(f"  Total pages: {total_pages}, chunk size: {chunk_size}, chunks: {len(chunks)}")

    merged: dict[str, set[int]] = {}

    def _run(chunk: tuple[int, int]) -> dict[str, list[int]]:
        start, end = chunk
        pages_range = list(range(start, end + 1))
        chunk_bytes = slice_pdf(pdf_bytes, pages_range)
        logger.info(f"  Chunk pages {start}-{end} ({len(chunk_bytes)} bytes)")
        return _detect_chunk(pd_config, chunk_bytes, start, end)

    with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as pool:
        futures = {pool.submit(_run, c): c for c in chunks}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                part = fut.result()
                for domain, pages in part.items():
                    merged.setdefault(domain, set()).update(pages)
                logger.info(f"  Chunk {c[0]}-{c[1]} -> { {k: sorted(v) for k, v in part.items()} }")
            except Exception as e:
                logger.error(f"  Chunk {c[0]}-{c[1]} FAILED: {e}", exc_info=True)

    cleaned = {domain: sorted(pages) for domain, pages in merged.items() if pages}

    (output_dir / "page_detection.json").write_text(json.dumps(cleaned, ensure_ascii=False, indent=2))
    logger.info(
        f"Page detection done in {time.time() - t0:.1f}s: "
        f"{len(cleaned)} domains, {sorted({p for v in cleaned.values() for p in v})} pages"
    )
    return cleaned


# -- OCR step --

def run_ocr_step(
    pdf_bytes: bytes,
    ocr_config: dict,
    output_dir: Path,
    pages: list[int] | None = None,
) -> list[dict]:
    logger.info("=== STEP: OCR ===")
    t0 = time.time()
    if pages:
        logger.info(f"  Restricting OCR to pages: {pages}")
    page_elements = call_ocr(
        pdf_bytes,
        model_name=ocr_config.get("model_name", "prebuilt-read"),
        pages=pages,
    )
    total_words = sum(len(pe.get("words", [])) for pe in page_elements)
    logger.info(f"OCR done in {time.time() - t0:.1f}s: {len(page_elements)} pages, {total_words} words")

    (output_dir / "ocr_results.json").write_text(json.dumps(page_elements, ensure_ascii=False, indent=2))
    return page_elements


# -- Single agent execution --

def run_agent(
    agent_config: dict,
    step_config: dict,
    pdf_bytes: bytes,
    page_elements: list[dict] | None,
    output_dir: Path,
) -> dict[str, Any]:
    domain_key = agent_config["domain_key"]
    logger.info(f"  Agent [{domain_key}] starting...")
    t0 = time.time()

    result = extract(
        pdf_bytes=pdf_bytes,
        system_prompt=agent_config.get("system_prompt") or step_config["system_prompt"],
        user_prompt=agent_config["user_prompt"],
        model_name=agent_config.get("model_name") or step_config["model_name"],
        page_elements=page_elements,
        response_schema=agent_config.get("response_schema"),
    )

    agent_dir = output_dir / "agents" / domain_key
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "result.json").write_text(json.dumps(result.data, ensure_ascii=False, indent=2))

    duration = time.time() - t0
    logger.info(f"  Agent [{domain_key}] done in {duration:.1f}s ({result.input_tokens} in / {result.output_tokens} out)")
    return {
        "domain_key": domain_key,
        "status": "COMPLETED",
        "result": result.data,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "duration": duration,
    }


# -- Data extraction step --

def _filter_pages(page_elements: list[dict] | None, pages: list[int] | None) -> list[dict] | None:
    if page_elements is None or not pages:
        return page_elements
    wanted = set(pages)
    return [pe for pe in page_elements if pe.get("page") in wanted]


def run_data_extraction_step(
    llm_config: dict,
    pdf_bytes: bytes,
    page_elements: list[dict] | None,
    output_dir: Path,
    page_detection: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    logger.info("=== STEP: DATA_EXTRACTION ===")
    executions = llm_config.get("executions")

    if not executions:
        # Single-call extraction
        logger.info("  Mode: single-call extraction")
        result = extract(
            pdf_bytes=pdf_bytes,
            system_prompt=llm_config["system_prompt"],
            user_prompt=llm_config["user_prompt"],
            model_name=llm_config["model_name"],
            page_elements=page_elements,
            response_schema=llm_config.get("response_schema"),
        )
        (output_dir / "final_result.json").write_text(json.dumps(result.data, ensure_ascii=False, indent=2))
        logger.info(f"  Done ({result.input_tokens} in / {result.output_tokens} out)")
        return {"result": result.data}

    # Multi-agent parallel extraction
    if page_detection is not None:
        active = [a for a in executions if page_detection.get(a["domain_key"])]
        skipped = [a["domain_key"] for a in executions if not page_detection.get(a["domain_key"])]
        if skipped:
            logger.info(f"  Skipping agents with 0 detected pages: {skipped}")
    else:
        active = list(executions)

    logger.info(f"  Mode: multi-agent parallel ({len(active)} agents)")
    merged: dict[str, Any] = {}
    metrics: list[dict] = []

    if not active:
        (output_dir / "final_result.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        return {"result": merged, "metrics": metrics}

    def _pages_for(agent: dict) -> list[int] | None:
        if page_detection is None:
            return None
        return page_detection.get(agent["domain_key"]) or None

    with ThreadPoolExecutor(max_workers=min(len(active), 5)) as pool:
        futures = {}
        for agent in active:
            pages = _pages_for(agent)
            agent_pdf = slice_pdf(pdf_bytes, pages) if pages else pdf_bytes
            agent_pe = _filter_pages(page_elements, pages)
            if pages:
                logger.info(f"  Agent [{agent['domain_key']}] pages={pages} ({len(agent_pdf)} bytes)")
            fut = pool.submit(run_agent, agent, llm_config, agent_pdf, agent_pe, output_dir)
            futures[fut] = agent["domain_key"]
        for future in as_completed(futures):
            dk = futures[future]
            try:
                r = future.result()
                merged[r["domain_key"]] = r["result"]
                metrics.append(r)
            except Exception as e:
                logger.error(f"  Agent [{dk}] FAILED: {e}", exc_info=True)
                merged[dk] = None
                metrics.append({"domain_key": dk, "status": "FAILED", "error": str(e)})

    (output_dir / "final_result.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    completed = sum(1 for m in metrics if m.get("status") == "COMPLETED")
    total_in = sum(m.get("input_tokens", 0) for m in metrics)
    total_out = sum(m.get("output_tokens", 0) for m in metrics)
    logger.info(f"  Merged {completed}/{len(executions)} agents ({total_in} in / {total_out} out)")
    return {"result": merged, "metrics": metrics}


# -- Main --

def run_flow(flow_config_path: str, input_pdf_path: str, output_dir: str) -> None:
    t_start = time.time()
    with open(flow_config_path) as f:
        config = json.load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pdf_bytes = Path(input_pdf_path).read_bytes()
    logger.info(f"Input: {input_pdf_path} ({len(pdf_bytes)} bytes)")
    logger.info(f"Flow: {config.get('flow_type', 'UNKNOWN')} with {len(config['steps'])} steps")

    page_elements: list[dict] | None = None
    page_detection: dict[str, list[int]] | None = None

    for step in sorted(config["steps"], key=lambda s: s["step_order"]):
        step_type = step["step_type"]

        if step_type == "PAGE_DETECTION":
            page_detection = run_page_detection_step(
                pdf_bytes, step["page_detection_step_config"], output
            )

        elif step_type == "OCR":
            pages = sorted({p for v in page_detection.values() for p in v}) if page_detection else None
            page_elements = run_ocr_step(pdf_bytes, step.get("ocr_step_config", {}), output, pages=pages)

        elif step_type == "DATA_EXTRACTION":
            run_data_extraction_step(
                step["llm_step_config"], pdf_bytes, page_elements, output, page_detection=page_detection
            )

        else:
            logger.warning(f"Unknown step type: {step_type}")

    logger.info(f"=== DONE in {time.time() - t_start:.1f}s === Results: {output}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    flow_config_path = os.environ.get("FLOW_CONFIG_PATH", str(repo_root / "flow_config.json"))
    input_pdf_path = os.environ.get("INPUT_PDF_PATH", str(repo_root / "data" / "input.pdf"))
    output_dir = os.environ.get("OUTPUT_DIR", str(repo_root / "data" / "output"))

    if not Path(input_pdf_path).exists():
        logger.error(f"Input PDF not found: {input_pdf_path}")
        logger.error("Place a PDF at data/input.pdf or set INPUT_PDF_PATH")
        sys.exit(1)

    run_flow(flow_config_path, input_pdf_path, output_dir)
