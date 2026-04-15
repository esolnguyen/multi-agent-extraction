"""Utilities for mapping LLM output to OCR bounding boxes.

Two paths:
- OCR path: map_wid_to_ocr_data() resolves word IDs to bboxes/confidence
- FILE_ONLY path: map_file_only_to_result() normalizes {v,bbox,page} to {value,bbox,conf,page}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _combine_bboxes(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return []
    x_coords = [c for bbox in bboxes for c in (bbox[0], bbox[2])]
    y_coords = [c for bbox in bboxes for c in (bbox[1], bbox[3])]
    if not x_coords or not y_coords:
        return []
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def enrich_value_with_wids(value: str, wids: list[int], word_id_map: dict[int, dict]) -> dict:
    if not wids:
        return {"value": value, "bbox": [], "conf": 0.0, "page": 0}
    word_data_list = [word_id_map[wid] for wid in wids if wid in word_id_map]
    if not word_data_list:
        return {"value": value, "bbox": [], "conf": 0.0, "page": 0}
    bboxes = [wd["box"] for wd in word_data_list]
    combined_bbox = _combine_bboxes(bboxes)
    avg_conf = sum(wd["conf"] for wd in word_data_list) / len(word_data_list)
    return {
        "value": value,
        "bbox": combined_bbox,
        "conf": round(avg_conf, 4),
        "page": word_data_list[0]["page"],
    }


def map_wid_to_ocr_data(llm_output: dict, page_elements: list[dict[str, Any]]) -> dict:
    """Map LLM output with word IDs to OCR bounding box data.

    page_elements: list of OcrPageElement-like dicts with 'words' and 'page' fields.
    Each word has: id, text, box (list[float]), conf (float).
    """
    word_id_map: dict[int, dict] = {}
    for pe in page_elements:
        for word in pe.get("words", []):
            text = (word.get("text") or "").strip()
            if text:
                word_id_map[word["id"]] = {
                    "text": text,
                    "box": [float(c) for c in word["box"]],
                    "conf": float(word.get("conf", 0)),
                    "page": int(pe["page"]),
                }

    def enrich_field(field_data):
        if not isinstance(field_data, dict):
            return field_data

        # Handle attributes list pattern
        if "attributes" in field_data and isinstance(field_data["attributes"], list):
            transformed = {}
            for attr in field_data["attributes"]:
                if isinstance(attr, dict) and "name" in attr and "value" in attr:
                    name = attr["name"]
                    value = attr["value"]
                    if isinstance(name, dict) and "v" in name:
                        key = name["v"]
                        value_wids = value.get("wids", []) if isinstance(value, dict) else []
                        transformed[key] = enrich_value_with_wids(
                            value.get("v", "") if isinstance(value, dict) else value,
                            value_wids, word_id_map,
                        )
            enriched = {}
            for key, value in field_data.items():
                if key != "attributes":
                    enriched[key] = _recurse(value)
            enriched.update(transformed)
            return enriched

        if "v" in field_data and "wids" in field_data:
            return enrich_value_with_wids(field_data["v"], field_data["wids"], word_id_map)

        return {k: _recurse(v) for k, v in field_data.items()}

    def _recurse(value):
        if isinstance(value, dict):
            return enrich_field(value)
        if isinstance(value, list):
            return [enrich_field(i) if isinstance(i, dict) else i for i in value]
        return value

    return enrich_field(llm_output)


def normalize_v_bbox(field_data: dict) -> dict:
    bbox_raw = field_data.get("bbox")
    bbox = []
    if isinstance(bbox_raw, dict):
        try:
            bbox = [float(bbox_raw.get(k, 0)) for k in ("x1", "y1", "x2", "y2")]
        except (ValueError, TypeError):
            pass
    elif isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        try:
            bbox = [float(c) for c in bbox_raw]
        except (ValueError, TypeError):
            pass
    return {
        "value": field_data.get("v", ""),
        "bbox": bbox,
        "conf": 0.0,
        "page": int(field_data.get("page", 0)),
    }


def map_file_only_to_result(llm_output: dict) -> dict:
    """Normalize FILE_ONLY LLM output {v, bbox, page} to {value, bbox, conf, page}."""

    def normalize_field(field_data):
        if not isinstance(field_data, dict):
            return field_data
        if "attributes" in field_data and isinstance(field_data["attributes"], list):
            transformed = {}
            for attr in field_data["attributes"]:
                if isinstance(attr, dict) and "name" in attr and "value" in attr:
                    name = attr["name"]
                    value = attr["value"]
                    if isinstance(name, dict) and "v" in name:
                        key = name["v"]
                        transformed[key] = (
                            normalize_v_bbox(value)
                            if isinstance(value, dict) and "v" in value
                            else normalize_field(value)
                        )
            normalized = {k: _recurse_norm(v) for k, v in field_data.items() if k != "attributes"}
            normalized.update(transformed)
            return normalized
        if "v" in field_data and "bbox" in field_data:
            return normalize_v_bbox(field_data)
        return {k: _recurse_norm(v) for k, v in field_data.items()}

    def _recurse_norm(value):
        if isinstance(value, dict):
            return normalize_field(value)
        if isinstance(value, list):
            return [normalize_field(i) if isinstance(i, dict) else i for i in value]
        return value

    return normalize_field(llm_output)
