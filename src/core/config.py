"""
Archon Configuration Module

Centralized configuration using pydantic-settings.
Loads from environment variables with validation.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Usage:
        from src.core.config import get_settings
        settings = get_settings()
        print(settings.openai_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="archon", description="Application name")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # -------------------------------------------------------------------------
    # API Server
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated allowed CORS origins"
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Azure OpenAI (optional)
    azure_openai_api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_deployment: str | None = Field(default=None, description="Azure OpenAI deployment name")

    # Google (optional)
    google_application_credentials: str | None = Field(default=None, description="Path to GCP credentials")
    gcp_project_id: str | None = Field(default=None, description="GCP project ID")

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for simple queries"
    )
    fallback_model: str = Field(
        default="claude-3-haiku-20240307",
        description="Fallback model if primary fails"
    )
    complex_model: str = Field(
        default="gpt-4o",
        description="Model for complex reasoning tasks"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model"
    )

    # -------------------------------------------------------------------------
    # Vector Database
    # -------------------------------------------------------------------------
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Chroma persistence directory (dev)"
    )
    pinecone_api_key: str | None = Field(default=None, description="Pinecone API key")
    pinecone_environment: str | None = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="archon", description="Pinecone index name")

    @property
    def use_pinecone(self) -> bool:
        """Check if Pinecone is configured for production use."""
        return bool(self.pinecone_api_key and self.app_env == "production")

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/archon",
        description="PostgreSQL connection URL"
    )

    @property
    def async_database_url(self) -> str:
        """Convert sync URL to async URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry OTLP endpoint"
    )
    otel_service_name: str = Field(default="archon", description="OTEL service name")

    # LangSmith (optional)
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: str | None = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="archon", description="LangSmith project name")

    # -------------------------------------------------------------------------
    # Cost Management
    # -------------------------------------------------------------------------
    daily_budget_usd: float = Field(default=10.0, description="Daily budget in USD")
    alert_threshold_percent: int = Field(default=80, description="Budget alert threshold %")

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------
    rate_limit_requests: int = Field(default=100, description="Max requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper

    @field_validator("daily_budget_usd")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        """Ensure budget is positive."""
        if v <= 0:
            raise ValueError("Daily budget must be positive")
        return v

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to avoid re-reading environment on every call.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# For convenience, export a settings instance
settings = get_settings()
