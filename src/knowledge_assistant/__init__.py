"""
University Knowledge Assistant SDK (Group 4 - NMIMS MPSTME)

    from knowledge_assistant import UniversityKnowledgeEngine
    engine = UniversityKnowledgeEngine()
    engine.search("minimum attendance requirement", top_k=3)
    engine.ask("What is the minimum attendance for exams?")
"""

from .client import EngineNotReady, UniversityKnowledgeEngine
from .config import EngineConfig
from .schemas import AssistantResponse, RetrievalMetadata, SearchResult

__all__ = ["UniversityKnowledgeEngine", "EngineConfig", "EngineNotReady",
           "SearchResult", "AssistantResponse", "RetrievalMetadata"]
__version__ = "1.0.0"
__author__ = "Group 4 (E020, E024, E025, E026, E028) - SNLP"
