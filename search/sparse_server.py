"""
A warm, always-on SPLADE encoder kept alive in the main app process,
reached by background-job subprocesses over a plain loopback socket.

Why this exists: get_sparse_encoder's cold load is genuinely expensive -
confirmed directly: ~7.5s importing the sentence_transformers/torch/
transformers stack, plus ~12s instantiating the model itself, even with
its weights already cached locally (~20s total, not a network download
cost). That ~20s used to get paid FRESH on every single background job -
search, chat, upload, investigation - because each one spawns its own OS
process (see ui/callbacks/background.py's DiskcacheManager), and nothing
loaded in one job's process survives into the next one's. Reported
directly as "significant slowdown."

The textbook fix is a persistent worker pool (Celery + a broker like
Redis) so a worker loads the model once and reuses it for every job it
ever handles after that - but this app ships to run on a user's own
machine via a plain `uv run python app.py`, with no Docker/Redis assumed
to be available there. So instead: preload the model exactly once, in the
one process that's ALREADY long-lived for the app's whole run (the main
Dash/Flask server - see app.py's __main__ guard), and let every spawned
background-job process reach into it over a local socket instead of
loading its own copy. multiprocessing.connection is pure stdlib -
loopback-only, no network exposure, nothing new to install - and Windows
gets a real TCP loopback socket here (not a named pipe), which behaves
identically across platforms.

Every call is best-effort: if the warm server isn't reachable for ANY
reason (app just started and the model isn't loaded yet, the server
process isn't running at all - e.g. a standalone script or test -, a
crashed connection mid-call), callers fall back to loading their own
model locally, exactly like before this module existed. Nothing hard-
depends on this being up.
"""

import logging
import socket
import threading
from multiprocessing.connection import Client, Listener
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_PORT = 8951
# Static and world-readable-in-source is fine here - this is a loopback-only
# socket, not a network service; the authkey just guards against a random
# OTHER local process accidentally connecting, not a real adversary.
_AUTHKEY = b"atelier-sparse-encoder-loopback"
_CONNECT_TIMEOUT_SECONDS = 0.5


def start_warm_encoder_server(model_name: str) -> None:
    """
    Starts a daemon thread that preloads `model_name` and then serves
    encode/score requests over a loopback socket, for the lifetime of the
    calling process. Returns immediately - the ~20s preload happens IN
    that background thread, not on the caller's - so this can be called
    from app.py right at startup without delaying the app's own first
    page load by 20s; background jobs launched during that warm-up window
    just fall back to loading their own model, exactly as if this server
    didn't exist yet, until it comes up.

    Call exactly once, from the real serving process only - see app.py's
    __main__ guard for why: Werkzeug's dev-reloader (debug=True) runs this
    whole module twice, once in a watcher parent that never serves real
    requests and once in the actual child - starting this in both would
    double-pay the ~20s cold load AND crash the second attempt trying to
    bind a port the first one already holds.
    """
    def _serve() -> None:
        from search.sparse_encoder import get_sparse_encoder  # deferred: see this module's own docstring on why torch shouldn't load until this thread actually runs

        logger.info(f"Preloading SPLADE model '{model_name}' for the warm encoder server...")
        try:
            get_sparse_encoder(model_name)
        except Exception as e:
            # The model itself is broken (corrupted/incomplete local cache,
            # missing dependency, out of memory, etc.) - the warm server
            # goes off entirely rather than starting a listener that could
            # never actually serve a working encode anyway. Every caller
            # already treats an unreachable warm server as "load the model
            # myself" (see call_warm_encoder's docstring), so this is a
            # clean, silent-to-the-user degradation, not a crash - it just
            # means every background job pays its own cold load instead of
            # none of them paying it, exactly like before this module
            # existed. Logged clearly here specifically so THIS root cause
            # (a broken model, not a busy port) is distinguishable from the
            # bind-failure branch below in the logs.
            logger.warning(f"SPLADE model '{model_name}' failed to load ({e}) - warm encoder server will not start; every background job will fall back to loading its own model.")
            return
        logger.info("SPLADE model warm - starting local encoder server for background jobs to reuse it.")

        try:
            listener = Listener((_HOST, _PORT), authkey=_AUTHKEY)
        except OSError as e:
            # Most likely something's already bound to this port (a
            # previous unclean shutdown's process still alive, or a rare
            # double-start despite the __main__ guard) - background jobs
            # simply fall back to loading their own model, same as if this
            # server never existed. Not fatal to the app either way.
            logger.warning(f"Warm encoder server couldn't bind {_HOST}:{_PORT} ({e}) - background jobs will fall back to loading their own model.")
            return
        while True:
            try:
                conn = listener.accept()
            except OSError:
                return  # listener closed - not expected to ever happen, but exit cleanly rather than spin
            threading.Thread(target=_handle_connection, args=(conn,), daemon=True).start()

    threading.Thread(target=_serve, daemon=True, name="sparse-encoder-server").start()


def _handle_connection(conn) -> None:
    from search.sparse_encoder import (
        get_sparse_encoder,
        _encode_query_dict,
        _encode_documents,
        _encode_documents_with_similarity,
    )
    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                return  # client closed its end - normal, one connection per call_warm_encoder invocation
            try:
                op, model_name, payload = request
                model = get_sparse_encoder(model_name)  # already warm/cached in THIS process - instant
                if op == "encode_query_dict":
                    result = _encode_query_dict(model, payload["text"])
                elif op == "encode_documents":
                    result = _encode_documents(model, payload["texts"], payload.get("top_k", 128))
                elif op == "encode_documents_with_similarity":
                    vectors, similarities = _encode_documents_with_similarity(
                        model, payload["query"], payload["texts"], payload.get("top_k", 128)
                    )
                    result = {"vectors": vectors, "similarities": similarities}
                else:
                    raise ValueError(f"Unknown op '{op}'")
                conn.send(("ok", result))
            except Exception as e:
                conn.send(("error", str(e)))
    finally:
        conn.close()


def call_warm_encoder(op: str, model_name: str, payload: Dict[str, Any]) -> Optional[Any]:
    """
    Best-effort RPC to the warm encoder server. Returns None on ANY
    failure - unreachable, timed out, errored mid-call - never raises, so
    every caller in search/sparse_encoder.py can treat None as "load the
    model locally instead," the exact same thing they did before this
    server existed.
    """
    # socket.setdefaulttimeout affects the plain TCP socket
    # multiprocessing.connection.Client opens under the hood for a
    # ('host', port) address - Client() itself has no timeout parameter,
    # so this is the only way to bound how long a connect attempt can
    # block if something's listening on the port but not accepting.
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_CONNECT_TIMEOUT_SECONDS)
    try:
        conn = Client((_HOST, _PORT), authkey=_AUTHKEY)
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)

    try:
        conn.send((op, model_name, payload))
        status, result = conn.recv()
        if status != "ok":
            logger.warning(f"Warm encoder server returned an error for '{op}': {result}")
            return None
        return result
    except Exception as e:
        logger.warning(f"Warm encoder server call failed for '{op}': {e} - falling back to a local model load.")
        return None
    finally:
        conn.close()
