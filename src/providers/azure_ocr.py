import logging
import os
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

_client: DocumentIntelligenceClient | None = None


def _get_client() -> DocumentIntelligenceClient:
    global _client
    if _client is None:
        _client = DocumentIntelligenceClient(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_DI_KEY"]),
        )
    return _client


def call_ocr(
    content: bytes,
    model_name: str = "prebuilt-read",
    pages: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run Azure Document Intelligence OCR on PDF bytes.

    Returns list of page elements:
    [{"page": int, "words": [{"id": int, "text": str, "box": [x1,y1,x2,y2], "conf": float}]}]

    If `pages` is provided, only those 1-based page numbers are analyzed.
    """
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model_id": model_name,
        "body": AnalyzeDocumentRequest(bytes_source=content),
    }
    if pages:
        kwargs["pages"] = ",".join(str(p) for p in sorted(set(pages)))

    poller = client.begin_analyze_document(**kwargs)
    result = poller.result()

    page_elements: list[dict[str, Any]] = []
    word_id = 0

    for page in result.pages:
        words = []
        page_num = page.page_number

        for word in (page.words or []):
            polygon = word.polygon or []
            if len(polygon) >= 8:
                # polygon is [x1,y1, x2,y2, x3,y3, x4,y4] - take bounding rect
                xs = [polygon[i] for i in range(0, len(polygon), 2)]
                ys = [polygon[i] for i in range(1, len(polygon), 2)]
                # Normalize to 0-1 using page dimensions
                w = page.width or 1
                h = page.height or 1
                box = [min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h]
            else:
                box = [0, 0, 0, 0]

            words.append({
                "id": word_id,
                "text": word.content,
                "box": box,
                "conf": word.confidence or 0.0,
            })
            word_id += 1

        page_elements.append({"page": page_num, "words": words})

    logger.info(f"OCR completed: {len(page_elements)} pages, {word_id} words")
    return page_elements
