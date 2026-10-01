"""
Shared test setup.

The automated tests must never call a real LLM or embedding API, even when a
real key is saved in .env. Real API calls are slow (rate limits, pacing delay),
cost quota, and give non repeatable results. LLM behaviour is tested with the
scripted fakes in tests/fakes.py instead. The live model is exercised only by
scripts/smoke_test_llm.py and evaluation/evaluate.py.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "LLM_BASE_URL", None)
    monkeypatch.setattr(config, "LLM_REQUEST_DELAY_SECONDS", 0.0)
    try:
        import streamlit as st

        st.cache_resource.clear()   # drop an LLM client or retriever cached by an earlier run
        st.cache_data.clear()
    except Exception:  # pragma: no cover
        pass
    yield
