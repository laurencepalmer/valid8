import asyncio
import platform
from typing import Optional

from backend.services.ai.base import AIProvider
from backend.config import get_settings


class MLXProvider(AIProvider):
    """AI provider using MLX-LM for Apple Silicon local inference."""

    def __init__(self):
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            raise RuntimeError(
                "MLX provider requires Apple Silicon (arm64 macOS). "
                "Use AI_PROVIDER=ollama for non-Apple hardware."
            )
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "mlx"

    def _load_model(self):
        """Lazy-load the MLX model on first use."""
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.settings.mlx_model)

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        response_format: Optional[dict] = None,
    ) -> str:
        self._load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, formatted_prompt, max_tokens, temperature
        )
        return result

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from mlx_lm import generate

        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "MLX provider does not support embeddings. "
            "Use USE_LOCAL_EMBEDDINGS=true (default) for local embeddings via sentence-transformers."
        )
