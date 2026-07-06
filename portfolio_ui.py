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
