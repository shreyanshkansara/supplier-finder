"""Text embeddings.

- GeminiEmbedder: the real one, used when GEMINI_API_KEY is set.
- HashEmbedder: deterministic, offline, zero-cost bag-of-words hashing. Used in tests, CI, and
  before you have an API key. It is also the "keyword baseline" you can compare against in eval.

Both return 1536-dim unit vectors so they fit the same pgvector column. Vendors store which
embedder produced their vector (embedding_model) so the two spaces are never mixed.

Retrieval is asymmetric (a short query vs a long vendor description), so embed() takes a `kind`:
queries and documents are formatted differently for the Gemini embedding model.
"""

import hashlib
import logging
import math
import re
from typing import Literal, Protocol

from app.config import Settings, get_settings
from app.services.ratelimit import limiter_for, paced_call

log = logging.getLogger(__name__)

Kind = Literal["query", "document"]


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str], kind: Kind = "document") -> list[list[float]]: ...


def l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class HashEmbedder:
    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.name = f"hash-bow-{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(v)

    async def embed(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        return [self._vec(t) for t in texts]


class GeminiEmbedder:
    """Gemini embeddings, truncated to `dim` dimensions and L2-normalised.

    gemini-embedding-2 takes the task as a text prefix rather than a task_type parameter:
      query:    "task: search result | query: {text}"
      document: "title: {title} | text: {text}"
    """

    BATCH = 100

    def __init__(self, model: str, dim: int, client=None, *, api_key: str | None = None,
                 retry_attempts: int = 4, timeout_s: float = 30.0, rpm: int = 10_000, max_wait_s: float = 90.0):
        if client is None:
            client = make_client(api_key, retry_attempts, timeout_s)
        self.client, self.model, self.dim = client, model, dim
        self.name = f"gemini:{model}:{dim}"
        self.limiter, self.max_wait_s = limiter_for(f"embed:{model}", rpm), max_wait_s

    @staticmethod
    def format(text: str, kind: Kind) -> str:
        if kind == "query":
            return f"task: search result | query: {text}"
        return f"title: none | text: {text}"

    async def embed(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = [self.format(t, kind) for t in texts[i : i + self.BATCH]]
            # One Content per text. A plain list of strings would be sent as ONE content with many
            # parts, which a multimodal embedding model turns into a single combined vector.
            contents = [types.Content(role="user", parts=[types.Part(text=t)]) for t in batch]
            resp = await paced_call(
                lambda contents=contents: self.client.aio.models.embed_content(
                    model=self.model, contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=self.dim)),
                self.limiter, self.max_wait_s)
            vectors = [l2_normalize(list(e.values)) for e in resp.embeddings]
            if len(vectors) != len(batch) or any(len(v) != self.dim for v in vectors):
                raise ValueError(f"expected {len(batch)} vectors of {self.dim} dims from {self.model}")
            out.extend(vectors)
        return out


def make_client(api_key: str | None, retry_attempts: int = 4, timeout_s: float = 30.0, **http_kwargs):
    """One place that builds the google-genai client, with retries on rate limits and 5xx."""
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=int(timeout_s * 1000),  # milliseconds
        # 429s are NOT retried here: the SDK's short backoff would spend more of the per-minute
        # quota. app.services.ratelimit paces calls and honours the API's own retry delay instead.
        retry_options=types.HttpRetryOptions(attempts=retry_attempts, http_status_codes=[500, 502, 503, 504]),
        **http_kwargs,
    ))


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.gemini_api_key:
        return GeminiEmbedder(s.gemini_embedding_model, s.embedding_dim, api_key=s.gemini_api_key,
                              retry_attempts=s.llm_retry_attempts, timeout_s=s.llm_timeout_s,
                              rpm=s.gemini_embed_rpm, max_wait_s=s.llm_max_rate_wait_s)
    log.warning("GEMINI_API_KEY not set: using the offline HashEmbedder")
    return HashEmbedder(s.embedding_dim)
