from backend.services.ai.base import AIProvider
from backend.services.ai.openai_provider import OpenAIProvider
from backend.services.ai.anthropic_provider import AnthropicProvider
from backend.services.ai.ollama_provider import OllamaProvider
from backend.config import get_settings


def get_ai_provider() -> AIProvider:
    settings = get_settings()
    provider_name = settings.ai_provider.lower()

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        return OpenAIProvider()
    elif provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        return AnthropicProvider()
    elif provider_name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown AI provider: {provider_name}")


__all__ = [
    "AIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "get_ai_provider",
]
