"""
Thin wrapper around the chat model.

Every agent talks to the LLM through `LLMClient`, so:
  * the model name, timeout and retries come from config / .env
  * every failure (no key, timeout, API error, malformed JSON) becomes one of
    two exceptions that the agents catch and turn into a graceful fallback
  * structured output is always parsed into a Pydantic model
"""

import time
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """No API key or the LLM package is missing."""


class LLMCallError(RuntimeError):
    """The LLM call failed: timeout, API error, or output that fails the schema."""


class LLMClient:
    def __init__(self, chat_model=None, model_name: str | None = None, api_key: str | None = None):
        if chat_model is None:
            key = api_key or config.OPENAI_API_KEY
            if not key:
                raise LLMUnavailableError(
                    "OPENAI_API_KEY is not set. Add it to the .env file. "
                    "Calculations still work without it."
                )
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:  # pragma: no cover
                raise LLMUnavailableError("langchain-openai is not installed.") from e

            chat_model = ChatOpenAI(
                model=model_name or config.MODEL_NAME,
                api_key=key,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT_SECONDS,
                max_retries=config.LLM_MAX_RETRIES,
                base_url=config.LLM_BASE_URL,
            )
        self.chat_model = chat_model

    def structured(self, schema: type[T], system: str, user: str) -> T:
        """Ask the LLM for output that must match `schema`.

        Attempt 1: LangChain structured output (forces the named function; works on OpenAI).
        Attempt 2: some OpenAI-compatible providers (e.g. free tiers) reject a *named* forced
        function, so we retry with tool_choice="required" and parse the tool call ourselves.
        """
        _pace()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            runnable = self.chat_model.with_structured_output(schema, method="function_calling")
            result = runnable.invoke(messages)
            if result is None:
                raise LLMCallError("The LLM returned no structured output.")
            if isinstance(result, dict):
                result = schema.model_validate(result)
            return result
        except Exception as first_error:  # noqa: BLE001
            if not _looks_like_request_rejection(first_error):
                raise LLMCallError(f"{type(first_error).__name__}: {first_error}") from first_error
        try:
            _pace()
            ai = self.chat_model.bind_tools([schema], tool_choice="required").invoke(messages)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                raise LLMCallError("The LLM returned no structured output.")
            return schema.model_validate(calls[0]["args"])
        except LLMCallError:
            raise
        except Exception as e:
            raise LLMCallError(f"{type(e).__name__}: {e}") from e

    def bind_tools(self, tools: list, **kwargs):
        return self.chat_model.bind_tools(tools, **kwargs)

    def invoke(self, messages: list[BaseMessage]):
        _pace()
        try:
            return self.chat_model.invoke(messages)
        except Exception as e:
            raise LLMCallError(f"{type(e).__name__}: {e}") from e


def _looks_like_request_rejection(e: Exception) -> bool:
    """A 400-type rejection of the request format (not a timeout, auth or rate limit error)."""
    text = f"{type(e).__name__} {e}".lower()
    return "badrequest" in text or "400" in text or "tool_choice" in text or "invalid_argument" in text


def _pace() -> None:
    """Optional pause between calls, to stay under free tier rate limits."""
    if config.LLM_REQUEST_DELAY_SECONDS > 0:
        time.sleep(config.LLM_REQUEST_DELAY_SECONDS)


def get_llm() -> LLMClient | None:
    """Return a client, or None if no API key is configured."""
    try:
        return LLMClient()
    except LLMUnavailableError:
        return None


def message_text(message) -> str:
    """Plain text of an AI message (content can be a string or a list of blocks)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
