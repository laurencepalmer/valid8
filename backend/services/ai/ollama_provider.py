from typing import Optional
import socket
import aiohttp
from backend.services.ai.base import AIProvider
from backend.config import get_settings


class OllamaProvider(AIProvider):
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.embedding_model = settings.ollama_embedding_model

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        response_format: Optional[dict] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": response_format if response_format else "json",
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        connector = aiohttp.TCPConnector(family=socket.AF_INET)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("response", "")

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=60, connect=30)
        connector = aiohttp.TCPConnector(family=socket.AF_INET)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "input": texts,
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("embeddings", [])
