"""
Local token estimation utilities (no API calls required).

- Text tokens via tiktoken (cl100k_base encoding).
- Vision tokens via OpenAI's high-res formula.
"""

import io
import math

import tiktoken


def estimate_text_tokens(text: str) -> int:
    """Estimate the number of tokens for a text string using cl100k_base."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_vision_tokens_for_images(image_bytes_list: list[bytes]) -> int:
    """Estimate vision token cost using OpenAI's high-res formula.

    Formula per image: 85 (base) + 170 × (number of 512×512 tiles).
    """
    from PIL import Image

    total = 0
    for img_bytes in image_bytes_list:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        tiles_x = math.ceil(w / 512)
        tiles_y = math.ceil(h / 512)
        total += 85 + 170 * tiles_x * tiles_y
    return total
