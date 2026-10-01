"""
TEST DOUBLES. Used only by the unit tests, never by the application.

They let us test agent LOGIC (fallbacks, cross checks, source filtering,
tool execution) without paying for API calls. Real LLM behaviour is measured
separately with scripts/smoke_test_llm.py and the Phase 9 evaluation.
"""

from agents.llm import LLMCallError


class FakeLLM:
    def __init__(self, responses: dict | None = None, fail: bool = False, chat_model=None):
        self.responses = responses or {}
        self.fail = fail
        self.chat_model = chat_model
        self.calls: list[tuple[str, str, str]] = []

    def structured(self, schema, system, user):
        self.calls.append((schema.__name__, system, user))
        if self.fail:
            raise LLMCallError("TimeoutError: simulated")
        if schema.__name__ not in self.responses:
            # behave like a real failed call, so agents use their fallback
            raise LLMCallError(f"No scripted response for {schema.__name__}")
        resp = self.responses[schema.__name__]
        return resp(user) if callable(resp) else resp

    def bind_tools(self, tools, **kwargs):
        return self.chat_model

    def invoke(self, messages):
        return self.chat_model.invoke(messages)


class ScriptedChatModel:
    """Returns pre written AI messages in order (e.g. a tool call, then a final answer)."""

    def __init__(self, messages):
        self.queue = list(messages)
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.queue.pop(0)


class StubRetriever:
    def __init__(self, chunks, fail: bool = False):
        self.chunks = chunks
        self.fail = fail

    def retrieve(self, query, k=4):
        if self.fail:
            raise RuntimeError("index not found")
        return self.chunks[:k]
