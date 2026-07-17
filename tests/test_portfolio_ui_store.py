"""Regression test: the saved-portfolio store must be per-session, not a
process-global cached resource.

The original bug wired _get_store() with @st.cache_resource, which shares one
LocalPortfolioStore (and its browser-localStorage bridge, including that bridge's
in-memory item cache) across *every* user session on the server. That leaked one
visitor's saved portfolios to everyone opening the site. The fix stores the
store in st.session_state, which is per-user.
"""
import sys
import types

import pytest


class _FakeSessionState(dict):
    """Mimics st.session_state's attribute/dict access enough for the test."""


def _install_fake_streamlit(monkeypatch, session_state):
    fake = types.ModuleType("streamlit")
    fake.session_state = session_state
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def test_get_store_is_per_session_not_globally_cached(monkeypatch):
    # A fresh session_state stands in for one user's session.
    session_state = _FakeSessionState()
    _install_fake_streamlit(monkeypatch, session_state)

    constructed = []

    class FakeStore:
        def __init__(self):
            constructed.append(self)

    import importlib
    import portfolio_ui
    importlib.reload(portfolio_ui)
    monkeypatch.setattr(portfolio_ui, "LocalPortfolioStore", FakeStore)

    # Within one session, repeated calls reuse the same instance (no re-fetch
    # from the browser every rerun) and it lives in session_state.
    first = portfolio_ui._get_store()
    second = portfolio_ui._get_store()
    assert first is second
    assert len(constructed) == 1
    assert session_state["_portfolio_store"] is first

    # A *different* session (new session_state) must get its OWN store — this is
    # the property @st.cache_resource violated.
    other_session = _FakeSessionState()
    portfolio_ui.st.session_state = other_session
    third = portfolio_ui._get_store()
    assert third is not first
    assert len(constructed) == 2
