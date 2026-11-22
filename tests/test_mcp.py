import asyncio
import pytest


def test_mcp_read():
    pytest.importorskip("fastmcp")
    from document_ai.mcp_tools import make_mcp, read_mcp

    text = asyncio.run(read_mcp(make_mcp({"doc1": "Synthetic document"}), "doc1"))
    assert "Synthetic document" in text
