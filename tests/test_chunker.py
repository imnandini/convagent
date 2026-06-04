import sys
import pytest
sys.path.append("convagent")

from chunker import chunk_text_file


def test_chunk_text_file_success():
    chunks = chunk_text_file("data/sample.txt")

    assert len(chunks) > 0
    assert isinstance(chunks, list)

    for chunk in chunks:
        assert isinstance(chunk, str)

def test_chunk_text_file_invalid_format():
    with pytest.raises(ValueError):
        chunk_text_file("data/sample.pdf")