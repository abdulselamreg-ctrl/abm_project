import os
import time
import logging
from typing import List, Optional, Union, Generator

from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nvidia_nim_chatbot")

API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")
DEFAULT_MODEL = os.getenv("NVIDIA_NIM_DEFAULT_MODEL", "openai/gpt-oss-120b")

AVAILABLE_MODELS = [
    "meta/llama-4-maverick",
    "minimax/minimax-m2.7",
    "mistralai/mistral-nemotron",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "mistralai/mistral-large-3-675b",
    "mistralai/mistral-small-3.1-24b-instruct",
    "nvidia/nemotron-3-nano-omni",
    "meta/llama-4-scout",
    "mistralai/mistral-medium-3",
    "nvidia/nemotron-nano-12b-v2-vl",
    "mistralai/mistral-large-3.1-675b-instruct",
    "deepseek-ai/deepseek-r1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "openai/gpt-oss-120b",
    "qwen/qwen2.5-72b-instruct",
    "meta/llama-3.1-70b-instruct",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "deepseek-ai/deepseek-coder-v2-lite-instruct",
    "qwen/qwen2.5-coder-32b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
]

_WORKING_MODELS = AVAILABLE_MODELS

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    timeout=30,
)

def get_available_models() -> List[str]:
    return list(_WORKING_MODELS)

def _call_with_retries(messages, model, stream=False):
    last_exception = None
    for attempt in range(1, 3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
            )
            return response
        except RateLimitError as e:
            logger.warning(f"Rate limit on {model}, attempt {attempt}")
            last_exception = e
            time.sleep(2 ** attempt)
        except APIConnectionError as e:
            logger.warning(f"Connection error on {model}, attempt {attempt}")
            last_exception = e
            time.sleep(2 ** attempt)
        except APIError as e:
            logger.error(f"API error on {model}: {e}")
            raise e
    raise last_exception or Exception("Max retries exceeded")

def get_llm_reply(
    user_message: str,
    model_name: Optional[str] = None,
    history: Optional[List[Union[dict, object]]] = None,
) -> str:
    if not API_KEY:
        return "Error: NVIDIA_NIM_API_KEY is missing."

    messages = [
        {
            "role": "system",
            "content": (
                "You are ABM Chatbot, a helpful and clear assistant. "
                "Always format your responses using Markdown with proper line breaks. "
                "Use headings (##), bullet points (-), numbered lists, and **bold** for emphasis. "
                "When you write lists, recipes, or code, separate each item with a new line. "
                "Keep answers natural and direct."
            )
        }
    ]
    if history:
        for item in history[-6:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                role = getattr(item, "role", "")
                content = getattr(item, "content", "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": user_message.strip()})

    candidates = []
    if model_name:
        candidates.append(model_name)
    for m in _WORKING_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_error = None
    for model in candidates:
        try:
            response = _call_with_retries(messages, model, stream=False)
            return response.choices[0].message.content or "No reply from model."
        except Exception as e:
            last_error = e
            logger.error(f"Model {model} failed: {e}")

    return f"All models failed. Last error: {last_error}"

def stream_llm_reply(
    user_message: str,
    model_name: Optional[str] = None,
    history: Optional[List[Union[dict, object]]] = None,
) -> Generator[str, None, None]:
    if not API_KEY:
        yield "Error: NVIDIA_NIM_API_KEY is missing."
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are ABM Chatbot, a helpful and clear assistant. "
                "Always format your responses using Markdown with proper line breaks. "
                "Use headings (##), bullet points (-), numbered lists, and **bold** for emphasis. "
                "When you write lists, recipes, or code, separate each item with a new line. "
                "Keep answers natural and direct."
            )
        }
    ]
    if history:
        for item in history[-6:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                role = getattr(item, "role", "")
                content = getattr(item, "content", "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": user_message.strip()})

    candidates = []
    if model_name:
        candidates.append(model_name)
    for m in _WORKING_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_error = None
    for model in candidates:
        try:
            stream = _call_with_retries(messages, model, stream=True)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return
        except Exception as e:
            last_error = e
            logger.error(f"Streaming model {model} failed: {e}")

    yield f"All models failed. Last error: {last_error}"