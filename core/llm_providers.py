"""
LLM API routing for OpenAI, Gemini, and Anthropic.

Each provider function accepts a unified interface and handles both
text-based and vision-based (image) requests.
"""

from __future__ import annotations

import base64


def call_openai(
    api_key: str,
    model: str,
    prompt: str,
    text_content: str | None = None,
    image_bytes_list: list[bytes] | None = None,
    json_mode: bool = False,
) -> str:
    """Send a request to OpenAI's Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    messages: list[dict] = []
    if text_content:
        messages.append({
            "role": "user",
            "content": f"{prompt}\n\n---\n\n{text_content}",
        })
    elif image_bytes_list:
        content_parts: list[dict] = [{"type": "text", "text": prompt}]
        for img in image_bytes_list:
            b64 = base64.b64encode(img).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })
        messages.append({"role": "user", "content": content_parts})

    kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    text_content: str | None = None,
    image_bytes_list: list[bytes] | None = None,
    json_mode: bool = False,
) -> str:
    """Send a request to Google's Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    parts: list = []
    if text_content:
        parts.append(types.Part.from_text(text=f"{prompt}\n\n---\n\n{text_content}"))
    elif image_bytes_list:
        parts.append(types.Part.from_text(text=prompt))
        for img in image_bytes_list:
            parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))

    config_kwargs: dict = {"temperature": 0}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return resp.text


def call_anthropic(
    api_key: str,
    model: str,
    prompt: str,
    text_content: str | None = None,
    image_bytes_list: list[bytes] | None = None,
    json_mode: bool = False,
) -> str:
    """Send a request to Anthropic's Messages API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    content_blocks: list[dict] = []
    if text_content:
        content_blocks.append({
            "type": "text",
            "text": f"{prompt}\n\n---\n\n{text_content}",
        })
    elif image_bytes_list:
        content_blocks.append({"type": "text", "text": prompt})
        for img in image_bytes_list:
            b64 = base64.b64encode(img).decode()
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })

    system_msg = ""
    if json_mode:
        system_msg = "You must respond with ONLY valid JSON. No markdown fences, no explanation."

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=system_msg if system_msg else anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": content_blocks}],
    )
    return resp.content[0].text


# Lookup table for dispatching by provider name.
PROVIDER_CALLERS = {
    "OpenAI": call_openai,
    "Gemini": call_gemini,
    "Anthropic": call_anthropic,
}
