import sys
from pathlib import Path

import pytest

# Make the project root importable so `convagent` resolves from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convagent.chunker import chunk_text_file


def test_chunk_text_file_success():
    chunks = chunk_text_file("data/sample.txt")

    assert len(chunks) > 0
    assert isinstance(chunks, list)

    for chunk in chunks:
        assert isinstance(chunk, str)

def test_chunk_text_file_invalid_format():
    with pytest.raises(ValueError):
        chunk_text_file("data/sample.pdf")
