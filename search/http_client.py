import requests
import time
import threading
from abc import ABC
from requests.adapters import HTTPAdapter
from typing import Optional
from urllib3.util.retry import Retry

from core.config import USER_AGENT
    
class HttpClient(ABC):
    """
    A wrapper around requests.Session to maintain headers, User-Agent, 
    and timeout settings across multiple API calls.
    """
    def __init__(self, timeout: int = 25, request_per_second: Optional[float] = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout

        # Configure robust retries so API hiccups don't crash the search.
        # backoff_factor=2 -> retries sleep ~2s, 4s, 8s, 16s (urllib3's
        # formula: backoff_factor * 2**(retry_number-1)) when a host doesn't
        # send a Retry-After header. Bumped from the original 3 retries/
        # factor=1 (~1s,2s,4s) after repeated Semantic Scholar 429s during
        # live testing - S2's unauthenticated tier shares one global rate
        # pool across every anonymous caller, so a single 429 there often
        # needs several seconds of backoff (not just one) before the shared
        # window resets. respect_retry_after_header still takes priority
        # over this formula whenever a host actually sends one.
        retries = Retry(
            total=5,
            connect=3,
            read=3,
            status=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=(["GET","HEAD"]),
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.request_per_second = request_per_second
        self._last_request_ts = 0.0
        self._lock = threading.Lock()

    def _throttle(self):
        if not self.request_per_second:
            return
        min_interval = 1.0 / self.request_per_second
        with self._lock:
            now = time.monotonic()
            wait = min_interval - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
        self._throttle()
        r = self.session.get(
            url.strip(),
            params=params,
            headers=headers,
            timeout=self.timeout
        )
        r.raise_for_status()
        return r

    def get_json(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
        """Executes a GET request and returns the JSON payload."""
        r = self._get(url, params=params, headers=headers)
        return r.json()

    def get_text(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> str:
        """Executes a GET request and returns the raw text response."""
        r = self._get(url, params=params, headers=headers)
        return r.text
    
    def close(self):
        self.session.close()
