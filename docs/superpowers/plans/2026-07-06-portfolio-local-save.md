# Browser-Local Portfolio Save/Reload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users name and save the currently loaded portfolio to their browser's localStorage, then reload it on a return visit — no accounts, passwords, or backend.

**Architecture:** Two new modules keep logic out of the ~2500-line `main.py`. `local_store.py` wraps the `streamlit-local-storage` component and stores all saved portfolios under a single localStorage key as a `name → payload` JSON map. `portfolio_ui.py` renders the sidebar save/load/delete panel. `main.py` gets three small hooks: render the panel, capture the current portfolio into a payload, and inject a loaded payload through the *existing* `load_and_validate_csv` pipeline so saved portfolios and uploads share one code path.

**Tech Stack:** Python, Streamlit, pandas, `streamlit-local-storage`, pytest.

## Global Constraints

- Python 3.13 (project runs on `cpython-313`).
- Deployment target: Streamlit Community Cloud (ephemeral filesystem — no local file persistence, which is why storage is browser-side).
- New runtime dependency limited to exactly one: `streamlit-local-storage`. Test-only: `pytest`.
- Payload `schema_version` is `1`.
- Single localStorage key namespace: `portivex_portfolios`.
- Follow existing conventions: modules at repo root (like `data_engine.py`), no `src/` dir. Tests go under `tests/`.
- Saved settings are exactly two keys: `benchmark` (default `"^GSPC"`) and `max_weight_pct` (default `15.0`). No others.
- Guest/upload flow must remain functionally unchanged.

---

### Task 1: Payload serialization helpers

Pure functions that convert between the in-app transactions DataFrame + settings and the JSON-serialisable payload dict. No Streamlit, no localStorage — fully unit-testable in isolation. This is the correctness-critical core (a saved portfolio must reload identically to an upload).

**Files:**
- Create: `payload.py`
- Test: `tests/test_payload.py`
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `build_payload(transactions: pd.DataFrame, settings: dict) -> dict` — returns `{"schema_version": 1, "transactions": [...records...], "settings": {...}}`. Dates in `transactions` serialised as ISO `YYYY-MM-DD` strings.
  - `payload_to_csv_buffer(payload: dict) -> io.StringIO` — writes the payload's transactions to an in-memory CSV (columns `Ticker,Date,Action,Quantity,Price`) suitable as input to `load_and_validate_csv`. Buffer is rewound to position 0.
  - `payload_settings(payload: dict) -> dict` — returns the `settings` sub-dict with defaults filled (`benchmark="^GSPC"`, `max_weight_pct=15.0`) for any missing keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty file), then `tests/test_payload.py`:

```python
import io
import pandas as pd
import payload


def _sample_transactions():
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT"],
        "Date": pd.to_datetime(["2023-01-05", "2023-02-10"]),
        "Action": ["Buy", "Buy"],
        "Quantity": [10, 5],
        "Price": [130.2, 252.75],
    })


def test_build_payload_shape_and_versions():
    p = payload.build_payload(_sample_transactions(), {"benchmark": "^NSEI", "max_weight_pct": 20.0})
    assert p["schema_version"] == payload.SCHEMA_VERSION == 1
    assert p["settings"] == {"benchmark": "^NSEI", "max_weight_pct": 20.0}
    assert len(p["transactions"]) == 2
    # dates serialised as ISO strings, not Timestamps
    assert p["transactions"][0]["Date"] == "2023-01-05"
    assert p["transactions"][0]["Ticker"] == "AAPL"


def test_build_payload_is_json_serialisable():
    import json
    p = payload.build_payload(_sample_transactions(), {})
    json.dumps(p)  # must not raise


def test_payload_to_csv_buffer_roundtrips_columns():
    p = payload.build_payload(_sample_transactions(), {})
    buf = payload.payload_to_csv_buffer(p)
    assert isinstance(buf, io.StringIO)
    df = pd.read_csv(buf)
    assert list(df.columns) == ["Ticker", "Date", "Action", "Quantity", "Price"]
    assert df.iloc[0]["Ticker"] == "AAPL"
    assert float(df.iloc[1]["Price"]) == 252.75


def test_payload_settings_fills_defaults():
    assert payload.payload_settings({"settings": {}}) == {"benchmark": "^GSPC", "max_weight_pct": 15.0}
    assert payload.payload_settings({"settings": {"benchmark": "^NSEI"}}) == {"benchmark": "^NSEI", "max_weight_pct": 15.0}
    assert payload.payload_settings({}) == {"benchmark": "^GSPC", "max_weight_pct": 15.0}


def test_price_column_optional():
    df = _sample_transactions().drop(columns=["Price"])
    p = payload.build_payload(df, {})
    buf = payload.payload_to_csv_buffer(p)
    out = pd.read_csv(buf)
    # Price column still emitted (empty) so downstream schema is stable
    assert "Price" in out.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_payload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'payload'`

- [ ] **Step 3: Write minimal implementation**

Create `payload.py`:

```python
"""Serialization between the in-app transactions DataFrame + settings and a
JSON-serialisable payload stored in the browser. No Streamlit or storage here.
"""
import io
import pandas as pd

SCHEMA_VERSION = 1

# The columns a saved portfolio round-trips through load_and_validate_csv.
_CSV_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price"]

_DEFAULT_SETTINGS = {"benchmark": "^GSPC", "max_weight_pct": 15.0}


def build_payload(transactions: pd.DataFrame, settings: dict) -> dict:
    df = transactions.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    records = df.to_dict(orient="records")
    clean_settings = {k: settings[k] for k in _DEFAULT_SETTINGS if k in settings}
    return {
        "schema_version": SCHEMA_VERSION,
        "transactions": records,
        "settings": clean_settings,
    }


def payload_to_csv_buffer(payload: dict) -> io.StringIO:
    df = pd.DataFrame(payload.get("transactions", []))
    # Ensure a stable column set (Price may be absent in older/price-less saves).
    for col in _CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_CSV_COLUMNS]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def payload_settings(payload: dict) -> dict:
    stored = (payload or {}).get("settings", {}) or {}
    merged = dict(_DEFAULT_SETTINGS)
    for k in _DEFAULT_SETTINGS:
        if k in stored and stored[k] is not None:
            merged[k] = stored[k]
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_payload.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add payload.py tests/__init__.py tests/test_payload.py
git commit -m "feat: add portfolio payload serialization helpers"
```

---

### Task 2: Local storage wrapper

Wraps the `streamlit-local-storage` component behind the backend-agnostic interface from the spec. All saved portfolios live under one localStorage key holding a JSON `name → payload` map. The component is injected so tests use an in-memory fake and never need a browser.

**Files:**
- Create: `local_store.py`
- Test: `tests/test_local_store.py`
- Modify: `requirements.txt` (add dependency)

**Interfaces:**
- Consumes: nothing from prior tasks (independent of `payload.py`).
- Produces a `LocalPortfolioStore` class:
  - `__init__(self, backend=None)` — `backend` is any object with `getItem(key)` and `setItem(key, value)` (JSON strings). Defaults to a real `streamlit_local_storage.LocalStorage()` when `None`.
  - `list_portfolios(self) -> list[str]` — saved names, sorted case-insensitively.
  - `save_portfolio(self, name: str, payload: dict) -> None` — insert or overwrite by exact name. Raises `ValueError` on empty/whitespace name.
  - `load_portfolio(self, name: str) -> dict | None` — payload or `None`.
  - `delete_portfolio(self, name: str) -> None` — no-op if absent.
  - Module constant `STORAGE_KEY = "portivex_portfolios"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_local_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_local_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_store'`

- [ ] **Step 3: Write minimal implementation**

Create `local_store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_local_store.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Add the dependency**

Add to `requirements.txt` (append after the last line):

```
streamlit-local-storage>=0.1.5
```

- [ ] **Step 6: Commit**

```bash
git add local_store.py tests/test_local_store.py requirements.txt
git commit -m "feat: add browser-local portfolio store"
```

---

### Task 3: Sidebar save/load/delete UI

Renders the "My Saved Portfolios" panel. Pure UI glue over Tasks 1–2. It reads the current portfolio from session state to build a save payload, and stages a chosen portfolio for the loader (Task 4) to pick up. Because this is Streamlit UI that depends on the live `st.session_state` and a browser component, it is verified manually in Task 5's smoke test rather than unit-tested — keep the logic thin and delegate all testable work to `payload.py`/`local_store.py`.

**Files:**
- Create: `portfolio_ui.py`

**Interfaces:**
- Consumes:
  - `payload.build_payload`, `payload.SCHEMA_VERSION` (Task 1).
  - `LocalPortfolioStore` (Task 2).
  - Session state written by `main.py`: `st.session_state["_portfolio_cache"]["transactions"]` (a DataFrame), `st.session_state.get("data_loaded")`, `st.session_state.get("benchmark")`, `st.session_state.get("max_weight_pct")`.
- Produces:
  - `render_portfolio_panel() -> None` — renders the sidebar panel.
  - On Load, sets `st.session_state["_pending_saved_load"] = {"name": name, "payload": payload}` and calls `st.rerun()`. Task 4 consumes this key.

- [ ] **Step 1: Write the implementation**

Create `portfolio_ui.py`:

```python
"""Sidebar panel: save the current portfolio to the browser, reload, or delete.
Thin UI over payload.py and local_store.py.
"""
import streamlit as st

import payload
from local_store import LocalPortfolioStore


@st.cache_resource
def _get_store() -> LocalPortfolioStore:
    return LocalPortfolioStore()


def render_portfolio_panel() -> None:
    store = _get_store()

    st.markdown('<span class="sidebar-group-label">My Saved Portfolios</span>',
                unsafe_allow_html=True)

    try:
        names = store.list_portfolios()
    except Exception:
        st.caption("Saving isn't available in this browser.")
        return

    # ── Load ───────────────────────────────────────────────
    if names:
        chosen = st.selectbox("Load a saved portfolio", names,
                              key="_saved_load_select", label_visibility="collapsed")
        cols = st.columns(2)
        if cols[0].button("Load", key="_saved_load_btn", use_container_width=True):
            loaded = store.load_portfolio(chosen)
            if loaded is not None:
                st.session_state["_pending_saved_load"] = {"name": chosen, "payload": loaded}
                st.rerun()
        if cols[1].button("Delete", key="_saved_del_btn", use_container_width=True):
            st.session_state["_confirm_delete"] = chosen
        if st.session_state.get("_confirm_delete") == chosen:
            st.warning(f"Delete '{chosen}'?")
            dc = st.columns(2)
            if dc[0].button("Yes, delete", key="_saved_del_yes", use_container_width=True):
                store.delete_portfolio(chosen)
                st.session_state.pop("_confirm_delete", None)
                st.rerun()
            if dc[1].button("Cancel", key="_saved_del_no", use_container_width=True):
                st.session_state.pop("_confirm_delete", None)
                st.rerun()
    else:
        st.caption("No saved portfolios yet.")

    # ── Save ───────────────────────────────────────────────
    data_loaded = st.session_state.get("data_loaded") and "_portfolio_cache" in st.session_state
    save_name = st.text_input("Save current as…", key="_saved_name_input",
                              placeholder="Name this portfolio", disabled=not data_loaded)
    if not data_loaded:
        st.caption("Load a portfolio first to save it.")
        return

    if st.button("Save", key="_saved_save_btn", use_container_width=True):
        name = (save_name or "").strip()
        if not name:
            st.error("Enter a name.")
            return
        # Overwrite confirmation
        existing = store.list_portfolios()
        if name in existing and st.session_state.get("_confirm_overwrite") != name:
            st.session_state["_confirm_overwrite"] = name
            st.warning(f"Overwrite '{name}'? Press Save again to confirm.")
            return
        transactions = st.session_state["_portfolio_cache"]["transactions"]
        settings = {
            "benchmark": st.session_state.get("benchmark", "^GSPC"),
            "max_weight_pct": st.session_state.get("max_weight_pct", 15.0),
        }
        try:
            store.save_portfolio(name, payload.build_payload(transactions, settings))
            st.session_state.pop("_confirm_overwrite", None)
            st.success(f"Saved '{name}'.")
        except Exception as e:
            st.error(f"Couldn't save: {e}")
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import portfolio_ui; print('ok')"`
Expected: prints `ok` (no import error). If `streamlit_local_storage` isn't installed yet, run `pip install -r requirements.txt` first.

- [ ] **Step 3: Commit**

```bash
git add portfolio_ui.py
git commit -m "feat: add saved-portfolios sidebar panel"
```

---

### Task 4: Wire load-injection and panel into main.py

Hook the panel into the sidebar and make a staged saved load flow through the existing `load_and_validate_csv` pipeline via a synthetic file id, so saved portfolios reuse the entire downstream path. A fresh upload takes precedence over a staged saved load.

**Files:**
- Modify: `main.py` (sidebar render near line 1266; load-injection near lines 1287 and 1355–1377)

**Interfaces:**
- Consumes: `render_portfolio_panel` (Task 3); `payload.payload_to_csv_buffer`, `payload.payload_settings` (Task 1); `st.session_state["_pending_saved_load"]` staged by Task 3.
- Produces: no new public API; establishes the runtime behavior that a loaded payload is indistinguishable from an upload downstream.

- [ ] **Step 1: Add imports and render the panel**

In `main.py`, add to the import block (near the other local-module imports around line 44):

```python
from portfolio_ui import render_portfolio_panel
import payload as _payload
```

Then, in the sidebar, immediately after the uploader line (`main.py:1266`):

```python
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    render_portfolio_panel()
```

- [ ] **Step 2: Resolve the effective data source (upload vs. saved load)**

Right after the sidebar `with` block ends and before the `if uploaded_file is None:` guard (currently `main.py:1287`), insert logic that turns a staged saved load into a file-like source. Replace the existing block starting at `if uploaded_file is None:` down to the `st.stop()` guard's opening so it accounts for a pending saved load.

Insert immediately **before** `if uploaded_file is None:` (line 1287):

```python
# ── Resolve effective portfolio source: a fresh upload always wins over
#    a staged saved-portfolio load. A staged load is turned into an
#    in-memory CSV so it flows through the exact same pipeline as an upload.
_pending = st.session_state.get("_pending_saved_load")
_saved_source = None
_saved_source_id = None
if uploaded_file is None and _pending is not None:
    _saved_source = _payload.payload_to_csv_buffer(_pending["payload"])
    _saved_source_id = f"saved:{_pending['name']}"
    # Apply saved settings before the pipeline reads them.
    _s = _payload.payload_settings(_pending["payload"])
    st.session_state["benchmark"] = _s["benchmark"]
    st.session_state["max_weight_pct"] = _s["max_weight_pct"]

_effective_source = uploaded_file if uploaded_file is not None else _saved_source
```

- [ ] **Step 3: Route the empty-state guard through the effective source**

Change the guard at `main.py:1287` from:

```python
if uploaded_file is None:
```

to:

```python
if _effective_source is None:
```

(The empty-state hero UI and `st.stop()` inside stay unchanged. Also clear the pending flag when nothing is loaded: inside this block, before `st.stop()`, add `st.session_state.pop("_pending_saved_load", None)`.)

- [ ] **Step 4: Use the effective source for the file id and load call**

Change the file-id line at `main.py:1355` from:

```python
_file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
```

to:

```python
_file_id = _saved_source_id if _saved_source_id is not None else getattr(uploaded_file, "file_id", uploaded_file.name)
```

Change the load call at `main.py:1377` from:

```python
    result       = load_and_validate_csv(uploaded_file)
```

to:

```python
    result       = load_and_validate_csv(_effective_source)
```

- [ ] **Step 5: Clear the pending flag after a successful load**

Immediately after `st.session_state["data_loaded"] = True` (currently `main.py:1478`), add:

```python
    st.session_state.pop("_pending_saved_load", None)
```

- [ ] **Step 6: Verify the app imports and starts**

Run: `python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read()); print('syntax ok')"`
Expected: prints `syntax ok`

Then run: `python -m streamlit run main.py --server.headless true --server.port 8599 & sleep 12 && curl -s -o /dev/null -w "%{http_code}" http://localhost:8599/ && kill %1`
Expected: prints `200` (app boots without error). If port is busy, use another port.

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: wire saved-portfolio load through the upload pipeline"
```

---

### Task 5: End-to-end payload round-trip test + manual smoke

Locks in the critical guarantee: a portfolio serialized to a payload and turned back into a CSV validates through `load_and_validate_csv` to an equivalent portfolio, with prices preserved so no network refetch is needed. Uses user-provided prices so the test needs no network.

**Files:**
- Test: `tests/test_payload_roundtrip.py`

**Interfaces:**
- Consumes: `payload.build_payload`, `payload.payload_to_csv_buffer` (Task 1); `data_engine.load_and_validate_csv`.
- Produces: nothing (test-only).

- [ ] **Step 1: Write the failing test**

Create `tests/test_payload_roundtrip.py`:

```python
import pandas as pd
import payload
from data_engine import load_and_validate_csv


def test_roundtrip_preserves_holdings_without_network():
    # All rows carry a user-provided Price, so load_and_validate_csv performs
    # no Yahoo fetch (fetch_historical_prices only fetches price-less rows).
    transactions = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "AAPL"],
        "Date": pd.to_datetime(["2023-01-05", "2023-02-10", "2024-06-01"]),
        "Action": ["Buy", "Buy", "Sell"],
        "Quantity": [10, 5, 3],
        "Price": [130.20, 252.75, 189.50],
    })
    p = payload.build_payload(transactions, {"benchmark": "^GSPC", "max_weight_pct": 15.0})
    buf = payload.payload_to_csv_buffer(p)

    result = load_and_validate_csv(buf)
    df = result[0] if isinstance(result, tuple) else result

    assert df is not None and not df.empty
    for col in ("Ticker", "Date", "Action", "Quantity", "Price"):
        assert col in df.columns
    # Same set of tickers survived
    assert set(df["Ticker"]) == {"AAPL", "MSFT"}
    # Prices preserved exactly (proves no refetch overwrote them)
    aapl_buy = df[(df["Ticker"] == "AAPL") & (df["Action"] == "Buy")].iloc[0]
    assert abs(float(aapl_buy["Price"]) - 130.20) < 1e-6
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_payload_roundtrip.py -v`
Expected: PASS (1 passed). If it fails because `load_and_validate_csv` attempts a network call, that means a row lacked a price — confirm all test rows have `Price` set.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (Tasks 1, 2, 5).

- [ ] **Step 4: Manual smoke test (browser)**

Run the app locally: `python -m streamlit run main.py`. Then:
1. Upload a small portfolio CSV → confirm it loads as today.
2. In the sidebar "My Saved Portfolios", type a name and click Save → success message.
3. Reload the browser tab (full refresh) → the portfolio does NOT auto-load (expected — guest default).
4. Select the saved name, click Load → the same portfolio renders, no re-upload.
5. Change benchmark, save under the same name → confirm "Overwrite?" prompt, then Save again → overwrites.
6. Click Delete → confirm prompt → portfolio disappears from the list.

Record the result. If any step fails, use systematic-debugging before proceeding.

- [ ] **Step 5: Commit**

```bash
git add tests/test_payload_roundtrip.py
git commit -m "test: end-to-end payload round-trip through the loader"
```

---

## Self-Review Notes

- **Spec coverage:** `local_store.py` (Task 2), `portfolio_ui.py` (Task 3), payload format (Task 1), load-flow reuse via synthetic id + upload precedence (Task 4), save/overwrite flow (Task 3), error handling for corrupt JSON / unavailable storage (Tasks 2–3), tests for store + round-trip + precedence (Tasks 2, 5), single new dependency (Task 2). All spec sections map to a task.
- **Upload-precedence** is enforced in Task 4 Step 2 (`uploaded_file is not None` wins).
- **Prices-preserved-on-reload** (avoiding a network refetch) is proven by Task 5 and is why the payload stores the post-fetch `Price` column.
- **Cache-busting** on switching sources works because `_file_id` becomes `saved:<name>` for saved loads (Task 4 Step 4), differing from any upload id and busting `_portfolio_cache` via the existing logic at `main.py:1356`.
- **No placeholders**: every code and command step is concrete.
