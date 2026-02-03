"""Configuration management for ATNF-Chat.

Uses pydantic-settings for type-safe environment variable loading.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API Configuration
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use",
    )

    # OpenRouter Configuration (free tier fallback)
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for free tier fallback",
    )
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash:free",
        description="Default free model on OpenRouter",
    )

    # Application Settings
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    api_host: str = Field(
        default="127.0.0.1",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        description="API server port",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Catalogue Settings
    catalogue_cache_dir: Path | None = Field(
        default=None,
        description="Path to cache catalogue data",
    )
    catalogue_force_refresh: bool = Field(
        default=False,
        description="Force catalogue refresh on startup",
    )

    # Rate Limiting
    max_api_calls_per_minute: int = Field(
        default=60,
        description="Maximum API calls per minute",
    )
    max_tokens_per_request: int = Field(
        default=4096,
        description="Maximum tokens per request",
    )

    # Local Model Fallback
    use_local_fallback: bool = Field(
        default=False,
        description="Enable local model fallback",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama3.3:70b",
        description="Local model name",
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
