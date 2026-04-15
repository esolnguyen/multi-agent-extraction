"""Core LLM execution for data extraction using Gemini.

Handles:
- Building prompts with OCR TOON encoding
- Calling Gemini
- Mapping results (wid-to-bbox for OCR path, normalize for FILE_ONLY path)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from providers.gemini_llm import call_gemini
from .prompts import (
    OCR_PROMPT_TEMPLATE,
    USER_INSTRUCTIONS_TEMPLATE,
    FIXED_SYSTEM_PROMPT,
    FIXED_SYSTEM_PROMPT_FILE_ONLY,
)
from .utils import map_wid_to_ocr_data, map_file_only_to_result

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    data: dict[str, Any] | list[dict[str, Any]]
    input_tokens: int
    output_tokens: int


def build_ocr_toon(page_elements: list[dict[str, Any]]) -> str:
    """Encode OCR data as compact JSON for the prompt."""
    words = []
    for pe in page_elements:
        for w in pe.get("words", []):
            text = (w.get("text") or "").strip()
            if text:
                words.append({
                    "id": w["id"], "text": text,
                    "x1": w["box"][0], "y1": w["box"][1],
                    "x2": w["box"][2], "y2": w["box"][3],
                })
    return json.dumps({"words": words}, ensure_ascii=False) if words else ""


def build_prompt(
    user_prompt: str,
    page_elements: list[dict[str, Any]] | None = None,
    user_instruction: str | None = None,
) -> str:
    parts = [user_prompt]
    if page_elements:
        toon = build_ocr_toon(page_elements)
        if toon:
            parts.append(OCR_PROMPT_TEMPLATE.format(ocr_toon=toon))
    if user_instruction:
        parts.append(USER_INSTRUCTIONS_TEMPLATE.format(user_instructions=user_instruction))
    return "\n".join(parts)


def build_system_prompt(base_prompt: str, has_ocr: bool) -> str:
    fixed = FIXED_SYSTEM_PROMPT if has_ocr else FIXED_SYSTEM_PROMPT_FILE_ONLY
    return f"{base_prompt}\n{fixed}"


def extract(
    pdf_bytes: bytes,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    page_elements: list[dict[str, Any]] | None = None,
    response_schema: dict | None = None,
    user_instruction: str | None = None,
) -> ExtractionResult:
    """Run a single LLM extraction call and map the results."""
    has_ocr = bool(page_elements)
    full_system = build_system_prompt(system_prompt, has_ocr)
    full_prompt = build_prompt(user_prompt, page_elements, user_instruction)

    response = call_gemini(
        model_name=model_name,
        system_prompt=full_system,
        user_prompt=full_prompt,
        file_bytes=pdf_bytes,
        response_schema=response_schema,
    )

    parsed = json.loads(response["content"])

    if has_ocr:
        final = [map_wid_to_ocr_data(i, page_elements) for i in parsed] if isinstance(parsed, list) else map_wid_to_ocr_data(parsed, page_elements)
    else:
        final = [map_file_only_to_result(i) for i in parsed] if isinstance(parsed, list) else map_file_only_to_result(parsed)

    return ExtractionResult(
        data=final,
        input_tokens=response.get("input_tokens", 0),
        output_tokens=response.get("output_tokens", 0),
    )
