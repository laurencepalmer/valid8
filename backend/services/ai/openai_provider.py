from typing import Optional
from openai import AsyncOpenAI
from backend.services.ai.base import AIProvider
from backend.config import get_settings


class OpenAIProvider(AIProvider):
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model

    @property
    def name(self) -> str:
        return "openai"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        response_format: Optional[dict] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": response_format,
                },
            }

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
