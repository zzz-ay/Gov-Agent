import os
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, check_config


PROVIDER_DEFAULTS = {
    "qwen": {
        "model_env": "QWEN_CHAT_MODEL",
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_model": "qwen-plus",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "model_env": "DEEPSEEK_CHAT_MODEL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_model": "deepseek-chat",
        "default_base_url": "https://api.deepseek.com",
    },
    "openai": {
        "model_env": "OPENAI_CHAT_MODEL",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4o-mini",
        "default_base_url": None,
    },
}


def normalize_provider(provider: Optional[str] = None) -> str:
    selected = (provider or LLM_PROVIDER or "qwen").strip().lower()
    if selected not in PROVIDER_DEFAULTS:
        return "qwen"
    return selected


def resolve_provider_config(provider: Optional[str] = None) -> Dict[str, Any]:
    selected = normalize_provider(provider)
    defaults = PROVIDER_DEFAULTS[selected]

    api_key = os.getenv(defaults["api_key_env"]) or OPENAI_API_KEY
    base_url = os.getenv(defaults["base_url_env"]) or OPENAI_BASE_URL or defaults["default_base_url"]
    model = os.getenv(defaults["model_env"]) or CHAT_MODEL or defaults["default_model"]

    if selected == "openai" and os.getenv(defaults["base_url_env"]) is None:
        base_url = None

    return {
        "provider": selected,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "api_key_env": defaults["api_key_env"],
        "base_url_env": defaults["base_url_env"],
        "model_env": defaults["model_env"],
    }


def get_chat_llm(
    temperature: float = 0.1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> ChatOpenAI:
    check_config()
    config = resolve_provider_config(provider)

    kwargs: Dict[str, Any] = {
        "model": model or config["model"],
        "api_key": config["api_key"],
        "temperature": temperature,
    }

    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]

    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


def get_model_runtime_info() -> Dict[str, Any]:
    config = resolve_provider_config()
    api_key = config.get("api_key") or ""
    masked_key = None

    if api_key:
        if len(api_key) <= 10:
            masked_key = api_key[:2] + "****"
        else:
            masked_key = api_key[:6] + "****" + api_key[-4:]

    return {
        "provider": config["provider"],
        "chat_model": config["model"],
        "base_url": config["base_url"],
        "api_key_masked": masked_key,
        "provider_env": {
            "api_key_env": config["api_key_env"],
            "base_url_env": config["base_url_env"],
            "model_env": config["model_env"],
        },
        "supported_providers": sorted(PROVIDER_DEFAULTS.keys()),
    }
