"""PaddleOCR integration, loaded only when OCR is requested."""

from functools import lru_cache


@lru_cache(maxsize=4)
def _ocr(language: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(lang=language, enable_mkldnn=False)


def extract_text(image_path: str, language: str = "en") -> str:
    """Extract text from an image; reading order depends on the OCR model."""
    pages = _ocr(language).predict(image_path)
    return "\n".join(text for page in pages for text in page.get("rec_texts", []))
