from typing import Optional
from anthropic import AsyncAnthropic
from backend.services.ai.base import AIProvider
from backend.config import get_settings


class AnthropicProvider(AIProvider):
    def __init__(self):
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide embedding API. "
            "Use local embeddings or OpenAI embeddings instead."
        )
