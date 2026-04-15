import logging
import os
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig, Part

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def call_gemini(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    file_bytes: bytes,
    response_schema: dict | None = None,
    config_options: dict | None = None,
) -> dict[str, Any]:
    """Call Gemini with a PDF file and text prompt.

    Returns: {"content": str (JSON), "input_tokens": int, "output_tokens": int}
    """
    client = _get_client()
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    config = GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
    )
    if response_schema:
        config.response_schema = response_schema
    if config_options:
        if config_options.get("temperature") is not None:
            config.temperature = float(config_options["temperature"])
        if config_options.get("max_output_tokens") is not None:
            config.max_output_tokens = config_options["max_output_tokens"]
        if config_options.get("top_p") is not None:
            config.top_p = float(config_options["top_p"])

    parts = [
        Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
        Part.from_text(text=user_prompt),
    ]

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0

    logger.info(f"Gemini response: {input_tokens} in / {output_tokens} out")
    return {
        "content": response.text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
