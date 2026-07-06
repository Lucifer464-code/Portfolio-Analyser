import json
import pytest
from local_store import LocalPortfolioStore, STORAGE_KEY


class FakeBackend:
    """In-memory stand-in for streamlit_local_storage.LocalStorage."""
    def __init__(self):
        self.store = {}

    def getItem(self, key):
        return self.store.get(key)

    def setItem(self, key, value):
        self.store[key] = value


def _store():
    return LocalPortfolioStore(backend=FakeBackend())


def test_empty_store_lists_nothing():
    assert _store().list_portfolios() == []


def test_save_then_list_and_load():
    s = _store()
    s.save_portfolio("My Fund", {"schema_version": 1, "a": 1})
    assert s.list_portfolios() == ["My Fund"]
    assert s.load_portfolio("My Fund") == {"schema_version": 1, "a": 1}


def test_save_overwrites_same_name():
    s = _store()
    s.save_portfolio("P", {"v": 1})
    s.save_portfolio("P", {"v": 2})
    assert s.list_portfolios() == ["P"]
    assert s.load_portfolio("P") == {"v": 2}


def test_list_is_sorted_case_insensitive():
    s = _store()
    s.save_portfolio("banana", {})
    s.save_portfolio("Apple", {})
    assert s.list_portfolios() == ["Apple", "banana"]


def test_load_missing_returns_none():
    assert _store().load_portfolio("nope") is None


def test_delete_removes_entry():
    s = _store()
    s.save_portfolio("P", {"v": 1})
    s.delete_portfolio("P")
    assert s.list_portfolios() == []
    s.delete_portfolio("P")  # no-op, must not raise


def test_empty_name_rejected():
    s = _store()
    with pytest.raises(ValueError):
        s.save_portfolio("   ", {"v": 1})


def test_persists_under_single_key_as_json():
    backend = FakeBackend()
    s = LocalPortfolioStore(backend=backend)
    s.save_portfolio("P", {"v": 1})
    raw = backend.getItem(STORAGE_KEY)
    assert json.loads(raw) == {"P": {"v": 1}}


def test_corrupt_json_treated_as_empty():
    backend = FakeBackend()
    backend.setItem(STORAGE_KEY, "{not valid json")
    s = LocalPortfolioStore(backend=backend)
    assert s.list_portfolios() == []  # does not raise
