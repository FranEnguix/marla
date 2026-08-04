"""Static NASimEmu knowledge base and deterministic lightweight RAG."""

from marla.knowledge.retriever import KnowledgeBase, KnowledgeRule, load_knowledge_base, retrieve_rules

__all__ = ["KnowledgeBase", "KnowledgeRule", "load_knowledge_base", "retrieve_rules"]
