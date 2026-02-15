#!/usr/bin/env python3

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Config(BaseSettings):
    """Configuration settings for the application."""

    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL for the local Ollama server.")
    ollama_model: str = Field(default="qwen2.5:14b", description="Default Ollama model.")
    logfire_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


config = Config()
