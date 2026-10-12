
from io import BytesIO

from docx import Document
from pypdf import PdfReader


def parse_document(filename: str, content: bytes) -> dict:
    """Extract text from supported document formats."""

    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "txt":
        text = content.decode("utf-8-sig", errors="replace")

        return {
            "document_type": "txt",
            "page_count": None,
            "text": text,
        }

    if extension == "pdf":
        reader = PdfReader(BytesIO(content))

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append({
                "page_number": page_number,
                "text": page_text,
            })

        return {
            "document_type": "pdf",
            "page_count": len(pages),
            "pages": pages,
            "text": "\n".join(page["text"] for page in pages),
        }

    if extension == "docx":
        document = Document(BytesIO(content))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return {
            "document_type": "docx",
            "page_count": None,
            "text": "\n".join(paragraphs),
        }

    raise ValueError(
        "Unsupported file type. Upload a .txt, .pdf, or .docx file."
    )
