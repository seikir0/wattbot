from wattbot.chunking import chunk_document
from wattbot.config import RetrievalConfig
from wattbot.schema import DocumentMeta

META = DocumentMeta(doc_id="doc1", title="Doc 1", url="https://example.org/doc1.pdf")


def test_page_numbers_tracked():
    pages = ["First page paragraph.", "Second page paragraph."]
    chunks = chunk_document(META, pages, RetrievalConfig(chunk_size=800, chunk_overlap=0))
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(c.doc_id == "doc1" for c in chunks)


def test_long_document_split_into_multiple_chunks():
    pages = ["\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(10))]
    chunks = chunk_document(META, pages, RetrievalConfig(chunk_size=500, chunk_overlap=100))
    assert len(chunks) > 1


def test_empty_document_gives_no_chunks():
    assert chunk_document(META, ["", "   "], RetrievalConfig()) == []
