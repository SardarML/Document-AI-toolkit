"""In-process MCP demo; does not expose an HTTP endpoint."""


def make_mcp(documents: dict[str, str]):
    """Register a read-only document lookup tool with FastMCP."""
    from fastmcp import FastMCP

    server = FastMCP("Document Tools")

    @server.tool
    def read_document(document_id: str) -> str:
        """Read a document by ID without modifying it."""
        return documents.get(document_id, "Not found")

    return server


async def read_mcp(server, document_id: str) -> str:
    """Invoke the MCP tool using an in-process client."""
    from fastmcp import Client

    async with Client(server) as client:
        result = await client.call_tool("read_document", {"document_id": document_id})
        return result.content[0].text
