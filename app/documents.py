from dataclasses import dataclass, field
from typing import Any

@dataclass
class SourceDocument:
    """A normalized document from any connected knowledge source."""

    tenant_id: str
    source: str
    document_id: str
    title: str
    text: str
    url: str | None = None
    last_modified: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: list[dict[str, Any]] = field(default_factory=list)
