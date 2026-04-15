import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from shared_llm.execution import extract
from providers.azure_ocr import call_ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_flow")


# -- OCR step --

def run_ocr_step(pdf_bytes: bytes, ocr_config: dict, output_dir: Path) -> list[dict]:
    logger.info("=== STEP: OCR ===")
    t0 = time.time()
    page_elements = call_ocr(pdf_bytes, model_name=ocr_config.get("model_name", "prebuilt-read"))
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

def run_data_extraction_step(
    llm_config: dict,
    pdf_bytes: bytes,
    page_elements: list[dict] | None,
    output_dir: Path,
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
    logger.info(f"  Mode: multi-agent parallel ({len(executions)} agents)")
    merged: dict[str, Any] = {}
    metrics: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(len(executions), 5)) as pool:
        futures = {
            pool.submit(run_agent, agent, llm_config, pdf_bytes, page_elements, output_dir): agent["domain_key"]
            for agent in executions
        }
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

    for step in sorted(config["steps"], key=lambda s: s["step_order"]):
        step_type = step["step_type"]

        if step_type == "OCR":
            page_elements = run_ocr_step(pdf_bytes, step.get("ocr_step_config", {}), output)

        elif step_type == "DATA_EXTRACTION":
            run_data_extraction_step(step["llm_step_config"], pdf_bytes, page_elements, output)

        elif step_type == "PAGE_DETECTION":
            logger.info("=== STEP: PAGE_DETECTION (skipped in local mode) ===")

        else:
            logger.warning(f"Unknown step type: {step_type}")

    logger.info(f"=== DONE in {time.time() - t_start:.1f}s === Results: {output}")


if __name__ == "__main__":
    flow_config_path = os.environ.get("FLOW_CONFIG_PATH", "docker/flow_config.json")
    input_pdf_path = os.environ.get("INPUT_PDF_PATH", "data/input.pdf")
    output_dir = os.environ.get("OUTPUT_DIR", "data/output")

    if not Path(input_pdf_path).exists():
        logger.error(f"Input PDF not found: {input_pdf_path}")
        logger.error("Place a PDF at data/input.pdf or set INPUT_PDF_PATH")
        sys.exit(1)

    run_flow(flow_config_path, input_pdf_path, output_dir)
