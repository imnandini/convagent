import sys
sys.path.append("convagent")

from chunker import chunk_text_file


def test_chunk_text_file():
    chunks = chunk_text_file("data/sample.txt")

    assert len(chunks) > 0
    assert isinstance(chunks, list)

    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0