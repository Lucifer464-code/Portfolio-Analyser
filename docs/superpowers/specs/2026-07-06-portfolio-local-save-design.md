# Design: Save & Reload Portfolios (Browser-Local)

**Date:** 2026-07-06
**Status:** Approved design — ready for implementation planning
**App:** Portivex — Streamlit portfolio analytics dashboard (deployed on Streamlit Community Cloud)

## Goal

Let users save the portfolio they're analysing so they don't have to re-upload the
same CSV every visit. On a return visit **in the same browser**, they can reload a
previously saved portfolio with one click.

Explicitly **not** in scope: user accounts, passwords, login, or any server-side
storage. There is no backend and no external service. Portfolios are stored in the
user's own browser.

## Chosen approach: browser-local storage

The portfolio (transaction rows + a small set of analysis settings) is stored in the
browser's `localStorage`. No data leaves the user's machine; there is no database,
no credentials, and nothing for the operator to secure or sign up for.

### Accepted tradeoffs (inherent to this approach)

1. **Per-device / per-browser.** A portfolio saved in laptop Chrome is not available
   in a phone browser, in Firefox, or after the user clears browsing data. There is
   no cross-device sync — that would require a backend, which is out of scope.
2. **No accounts.** There is no user identity concept. The model is simply "this
   browser remembers your saved portfolios."
3. **Streamlit bridge required.** Streamlit executes the UI server-side and cannot
   read/write `localStorage` directly. A small third-party component bridges this.
   This is the only new dependency. See Dependencies.

These are stated so future maintainers understand they are consequences of the
"no backend, no passwords" decision — not defects to be fixed.

## Architecture

`main.py` remains the orchestrator. Two new focused modules are added, matching the
existing `*_engine.py` convention of keeping logic out of the ~2500-line UI file.

### `local_store.py` — storage layer

The only module that talks to `localStorage`. Wraps the bridge component and exposes
a small, backend-agnostic interface. All saved portfolios live under **one**
namespaced `localStorage` key (e.g. `portivex_portfolios`) holding a JSON object that
maps `name → payload`.

```
list_portfolios() -> list[str]                 # saved portfolio names, sorted
save_portfolio(name: str, payload: dict) -> None   # insert or overwrite by name
load_portfolio(name: str) -> dict | None       # returns payload, or None if absent
delete_portfolio(name: str) -> None
```

Rationale for a single key holding a name→payload map: one read/write per operation,
trivial to enumerate names, and no key-collision bookkeeping. The interface is
deliberately backend-agnostic so the storage mechanism could later change without
touching UI code.

### `portfolio_ui.py` — sidebar UI

Renders the "My Saved Portfolios" panel in the sidebar. No login UI.

- **Save current** — a name text field + Save button. Captures the currently loaded
  transactions + settings into a payload (see Payload) and calls
  `save_portfolio(name, payload)`. If the name already exists, it **overwrites** that
  entry after a small inline "Overwrite <name>?" confirm. Save is disabled when no
  portfolio is currently loaded.
- **Load** — a dropdown of saved names + Load button. On load, sets the payload into
  session state for injection (see Load flow) and reruns.
- **Delete** — removes the selected saved portfolio after an inline confirm.

### `main.py` — integration

Three small hooks, no changes to the analytics pipeline:

1. **Render the panel** in the sidebar, below the existing CSV uploader.
2. **Capture** — build the save payload from currently loaded data (see Payload).
3. **Inject on load** — feed a loaded payload's transactions into the *existing*
   `load_and_validate_csv` pipeline (see Load flow), so saved portfolios and uploads
   share one code path.

## Payload format

A saved portfolio is a single JSON-serialisable dict:

```json
{
  "schema_version": 1,
  "transactions": [
    {"Ticker": "AAPL", "Date": "2023-01-05", "Action": "Buy", "Quantity": 10, "Price": 130.2}
  ],
  "settings": {
    "benchmark": "^GSPC",
    "max_weight_pct": 15.0
  }
}
```

- **`transactions`** — the raw transaction rows as originally validated
  (`st.session_state["_portfolio_cache"]["transactions"]`), converted via
  `df.to_dict(orient="records")`. Dates serialised as ISO `YYYY-MM-DD` strings.
- **`settings`** — the analysis settings that persist in `st.session_state`:
  `benchmark` (default `^GSPC`) and `max_weight_pct` (default `15.0`). These are the
  two settings the app actually reads back from session state; other options use
  fixed defaults and are intentionally excluded (YAGNI).
- **`schema_version`** — guards against future payload changes. Loader checks it and
  ignores/upgrades unknown-but-compatible payloads gracefully.

## Load flow (reusing the existing pipeline)

The app currently keys its heavy computation cache off the uploaded file id
(`_last_file_id` at `main.py:1355`) and loads via `load_and_validate_csv(uploaded_file)`
at `main.py:1377`. To avoid a parallel, drift-prone code path, a loaded portfolio
flows through that **same** path:

1. On Load, `portfolio_ui` stores the chosen payload in
   `st.session_state["_pending_saved_load"]` and reruns.
2. Near the top of the data-load section, `main.py` checks for a pending saved load.
   If present, it serialises the payload's `transactions` into an in-memory CSV
   buffer (`io.StringIO`) and uses that as the input to `load_and_validate_csv`,
   exactly as if it were an uploaded file.
3. It assigns a **synthetic stable id** for the cache key, e.g.
   `saved:<name>` (or `saved:<name>:<hash-of-transactions>`), used in place of
   `uploaded_file.file_id`. This makes cache-busting work correctly when switching
   between a saved portfolio and a fresh upload, and when reloading after an
   overwrite.
4. The payload's `settings` are written into `st.session_state` (`benchmark`,
   `max_weight_pct`) before the pipeline reads them.
5. The pending flag is cleared so subsequent reruns don't re-trigger the load.

Precedence: a **freshly uploaded file takes priority** over a pending saved load, so
an explicit new upload is never silently overridden by a stale saved selection.

Everything downstream (validation, aggregation, risk analytics, tabs) runs unchanged
because the input is indistinguishable from an upload.

## Save flow

1. Save is available only when `st.session_state.get("data_loaded")` is true and a
   `_portfolio_cache` exists.
2. Build the payload from `_portfolio_cache["transactions"]` and current settings.
3. If the entered name collides with an existing saved name, show an inline
   "Overwrite?" confirm; on confirm, `save_portfolio` replaces that entry.
4. Show a brief success toast/message.

`localStorage` has a practical ~5MB per-origin budget; a normal portfolio (hundreds
of transactions) is a few KB. A guard in `save_portfolio` surfaces a friendly error
if a write ever fails (e.g. quota), rather than failing silently.

## Error handling

- **Bridge/component unavailable or storage blocked** (e.g. privacy mode): the panel
  degrades gracefully — it shows a short "saving isn't available in this browser"
  note and the app otherwise works exactly as today. The uploader path is never
  affected.
- **Corrupt/unparseable stored JSON:** treated as "no saved portfolios" rather than
  crashing; a reset is offered.
- **Payload missing required transaction columns on load:** the existing
  `load_and_validate_csv` validation already errors clearly; the same messaging is
  reused.
- **Empty name / whitespace-only name on save:** rejected inline.

## Testing

- **`local_store.py`** — unit-test the name→payload map logic (insert, overwrite,
  list ordering, delete, load-missing) against a fake/in-memory storage stub so tests
  don't require a browser. The bridge component is injected/mockable.
- **Payload round-trip** — a transactions DataFrame → payload → `StringIO` CSV →
  `load_and_validate_csv` produces an equivalent validated DataFrame. This is the
  critical correctness test: saved portfolios must reload identically to uploads.
- **Load precedence** — a fresh upload takes priority over a pending saved load.
- **Cache-busting** — switching between two saved portfolios (and between a saved
  portfolio and an upload) changes the synthetic id and busts `_portfolio_cache`.

## Dependencies

Add to `requirements.txt`:

- `streamlit-local-storage` — bridges Streamlit ↔ browser `localStorage`. Only new
  dependency. (Exact component pinned during implementation.)

No secrets, no service accounts, no database, no `.streamlit/secrets.toml` changes.

## Files touched

- **New:** `local_store.py`, `portfolio_ui.py`
- **Modified:** `main.py` (sidebar panel render + save capture + load injection near
  the existing data-load block), `requirements.txt`
- **New tests:** `tests/test_local_store.py`, `tests/test_payload_roundtrip.py`
  (or the project's chosen test location)
