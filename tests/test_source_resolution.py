"""A sticky file_uploader must not swallow a staged saved-portfolio load.

Streamlit's file_uploader returns the same object on every rerun until the
user clears it, so "is an upload present?" cannot distinguish a fresh upload
from the one that is merely still sitting in the widget.
"""
import payload


def test_fresh_upload_wins_over_pending_load():
    src, sid = payload.resolve_source(
        upload_id="a.csv", pending_name="B", last_upload_id=None
    )
    assert src == "upload"
    assert sid == "upload:a.csv"


def test_upload_with_no_pending_load_is_the_source():
    src, sid = payload.resolve_source(
        upload_id="a.csv", pending_name=None, last_upload_id="a.csv"
    )
    assert src == "upload"
    assert sid == "upload:a.csv"


def test_stale_upload_does_not_block_pending_load():
    """The bug: uploader still holds a.csv from earlier, user clicks Load on B."""
    src, sid = payload.resolve_source(
        upload_id="a.csv", pending_name="B", last_upload_id="a.csv"
    )
    assert src == "saved"
    assert sid == "saved:B"


def test_pending_load_with_empty_uploader():
    src, sid = payload.resolve_source(
        upload_id=None, pending_name="B", last_upload_id="a.csv"
    )
    assert src == "saved"
    assert sid == "saved:B"


def test_reupload_of_same_name_after_saved_load_is_fresh():
    """Cleared then re-uploaded: file_id differs, so it is a fresh upload."""
    src, sid = payload.resolve_source(
        upload_id="a.csv#2", pending_name=None, last_upload_id="a.csv"
    )
    assert src == "upload"
    assert sid == "upload:a.csv#2"


def test_nothing_at_all():
    assert payload.resolve_source(
        upload_id=None, pending_name=None, last_upload_id=None
    ) == (None, None)


def test_saved_portfolio_stays_active_while_stale_upload_lingers():
    """After a saved load consumes the pending flag, the still-present upload
    must not snap the app back to the uploaded file on the next rerun."""
    src, sid = payload.resolve_source(
        upload_id="a.csv", pending_name=None,
        last_upload_id="a.csv", active_saved_name="B",
    )
    assert src == "saved"
    assert sid == "saved:B"


def test_fresh_upload_overrides_active_saved_portfolio():
    """Uploading a genuinely new file while a saved portfolio is active wins."""
    src, sid = payload.resolve_source(
        upload_id="c.csv", pending_name=None,
        last_upload_id="a.csv", active_saved_name="B",
    )
    assert src == "upload"
    assert sid == "upload:c.csv"


def test_switching_between_two_saved_portfolios():
    """The reported bug: A active, uploader sticky, user clicks Load on B."""
    src, sid = payload.resolve_source(
        upload_id="a.csv", pending_name="B",
        last_upload_id="a.csv", active_saved_name="A",
    )
    assert src == "saved"
    assert sid == "saved:B"
