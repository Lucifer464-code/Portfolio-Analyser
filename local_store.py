"""Browser-local persistence for saved portfolios.

All saved portfolios live under a single localStorage key as a JSON object
mapping name -> payload. The component is injected so this is unit-testable
without a browser.
"""
import json

STORAGE_KEY = "portivex_portfolios"


class LocalPortfolioStore:
    def __init__(self, backend=None):
        if backend is None:
            from streamlit_local_storage import LocalStorage
            backend = LocalStorage()
        self._backend = backend

    def _read_map(self) -> dict:
        raw = self._backend.getItem(STORAGE_KEY)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    def _write_map(self, data: dict) -> None:
        self._backend.setItem(STORAGE_KEY, json.dumps(data))

    def list_portfolios(self) -> list:
        return sorted(self._read_map().keys(), key=str.lower)

    def save_portfolio(self, name: str, payload: dict) -> None:
        if not name or not name.strip():
            raise ValueError("Portfolio name cannot be empty.")
        data = self._read_map()
        data[name.strip()] = payload
        self._write_map(data)

    def load_portfolio(self, name: str):
        return self._read_map().get(name)

    def delete_portfolio(self, name: str) -> None:
        data = self._read_map()
        if name in data:
            del data[name]
            self._write_map(data)
