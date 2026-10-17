"""Backward-compatible alias: the engine now lives in `knowledge_assistant.client`."""

from .client import EngineNotReady, UniversityKnowledgeEngine  # noqa: F401
from .schemas import AssistantResponse, RetrievalMetadata, SearchResult  # noqa: F401
