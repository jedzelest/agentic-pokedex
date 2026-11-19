"""Turn text into embedding vectors (lists of numbers that capture meaning)."""

import openai

from agentic_pokedex.config import RagSettings

BATCH_SIZE = 100  # texts per API request


class Embedder:
    def __init__(self, settings: RagSettings):
        self.settings = settings
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY; separate from the chat LLM

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts[start:start + BATCH_SIZE],
                dimensions=self.settings.embedding_dimensions,  # must match vector(1024)
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
