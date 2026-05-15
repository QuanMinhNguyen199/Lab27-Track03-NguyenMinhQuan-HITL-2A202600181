"""LLM factory. Returns an Ollama-backed chat model."""

import os

from langchain_ollama import ChatOllama


def get_llm(temperature: float = 0.2) -> ChatOllama:
    model = os.environ.get("OLLAMA_MODEL", "llama2")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
