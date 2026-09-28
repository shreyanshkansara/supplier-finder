"""The LLM layer. It writes text and fills typed models. It never makes pass/fail decisions.

One job:
  parse_requirement(text)  -> SourcingRequirement   (structured output, then cleaned up in code)

Implementations:
  GeminiLLM     Gemini generate_content with a JSON schema from the Pydantic model
                (`response_json_schema`), validated back into the model by Pydantic. Used when
                GEMINI_API_KEY and GEMINI_MODEL are set.
  RuleBasedLLM  Regex parser + template. No key, no cost, deterministic. Also the fallback
                whenever a Gemini call fails, so a flaky API or a rate limit never breaks a run.

Numbers are never trusted to the model: units are normalised, absolute dates are turned into
days by code, and the delivery state comes from a lookup table.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.catalog import Category, Unit, norm_unit
from app.config import Settings, get_settings
from app.schemas import SourcingRequirement
from app.services.geo import CITY_TO_STATE, state_for
from app.services.ratelimit import limiter_for, paced_call

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
# What the model fills in. Kept separate from SourcingRequirement on purpose: the schema sent
# to the model has every field required-but-nullable and no numeric constraints, and all the
# real validation happens afterwards in code.
# ----------------------------------------------------------------------------------------
class ParsedRequirement(BaseModel):
    item: str | None = Field(description="The thing to buy, short, e.g. 'enamel paint'. null if not stated.")
    specs: str | None = Field(description="Every spec the buyer stated (sizes, thickness, density, material, "
                                          "finish, moisture...). Copy numbers exactly. null if none.")
    quantity: float | None = Field(description="Numeric quantity exactly as stated. null if not stated.")
    unit: Unit | None = Field(description="Unit of the quantity. Pairs, boxes, pieces, sets, rolls, nos -> 'pcs'; "
                                             "jobs or batches of work -> 'jobs'. null if not stated.")
    delivery_location: str | None = Field(description="Delivery city as written. null if not stated.")
    deadline_days: int | None = Field(description="Only for relative deadlines ('within 3 weeks' -> 21). "
                                                  "null otherwise.")
    deadline_date: str | None = Field(description="Only for absolute deadlines, as YYYY-MM-DD. null otherwise.")
    category: Category | None = Field(description="The supplier category this item belongs to. null if unclear.")
    notes: str | None = Field(description="Anything else relevant (e.g. a second item, 'urgent'). null if none.")


PARSE_INSTRUCTIONS = """You extract a purchase requirement for {company}, a furniture manufacturer in Jodhpur, India.
Today is {today}.

Rules:
- Extract ONLY what the buyer actually said. If something is not stated, use null. Never guess or fill defaults.
- Copy numbers exactly (quantities, thickness, density, moisture). Do not convert or round.
- A vague request ("some paint", "screws, urgent") must leave quantity, unit, deadline and location null.
- "urgent" or "ASAP" is not a deadline: put it in notes.
- If the buyer asks for more than one different item, extract the first and list the rest in notes.
"""

@dataclass
class LLMResult:
    """What every LLM call returns alongside its value, for audit and cost tracking."""

    source: str  # "gemini:<model>" | "rules" | "rules(fallback)"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def usage(self, step: str) -> dict:
        return {"step": step, "source": self.source, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "latency_ms": self.latency_ms, "error": self.error}


class LLM(Protocol):
    name: str

    async def parse_requirement(self, text: str) -> tuple[SourcingRequirement, LLMResult]: ...


# ----------------------------------------------------------------------------------------
# Code-side cleanup (applies to any parser)
# ----------------------------------------------------------------------------------------
def to_requirement(p: ParsedRequirement, today: date | None = None) -> SourcingRequirement:
    today = today or date.today()
    days = p.deadline_days
    if days is None and p.deadline_date:
        try:
            days = (date.fromisoformat(p.deadline_date) - today).days
        except ValueError:
            days = None
    data = {
        "item": (p.item or "").strip() or None,
        "specs": (p.specs or "").strip() or None,
        "quantity": p.quantity if p.quantity and p.quantity > 0 else None,
        "unit": norm_unit(p.unit),
        "delivery_location": (p.delivery_location or "").strip() or None,
        "delivery_state": state_for(p.delivery_location),
        "deadline_days": days if days and days > 0 else None,
        "category": p.category,
        "notes": p.notes,
    }
    try:
        return SourcingRequirement(**data)
    except ValidationError:  # belt and braces: drop whatever still fails validation
        return SourcingRequirement(item=data["item"], specs=data["specs"], notes=data["notes"])


# ----------------------------------------------------------------------------------------
# Rule-based implementation
# ----------------------------------------------------------------------------------------
_QTY = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(cft|sheets?|pcs|pieces?|pairs?|kgs?|kg|sq\s?ft|metres?|meters?|m\b|"
    r"litres?|liters?|l\b|boxes|box|rolls?|sets?)", re.I)
# "2,000 soft-close concealed hinges", "400 SS legs", "1 job of CNC routing": a count, then a
# countable noun within a few words. Tried only when _QTY (number directly followed by a unit) fails.
_COUNT = re.compile(r"(\d[\d,]*)\s+(?:[\w-]+\s+){0,4}?(hinges?|lifts?|legs?|boxes|box|sets?|pairs?|rolls?|"
                    r"handles?|channels?|frames?|bases?|pieces?|jobs?)\b", re.I)
_DEADLINE = re.compile(r"(?:within|in|by)\s+(\d+|a|one|two|three|four)\s+(days?|weeks?|months?)", re.I)
_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4}


def rule_based_parse(text: str) -> SourcingRequirement:
    req = SourcingRequirement(item=text.strip().rstrip("."), specs=text.strip())
    if m := _QTY.search(text):
        req.quantity = float(m.group(1).replace(",", ""))
        req.unit = norm_unit(m.group(2))
    elif m := _COUNT.search(text):
        req.quantity = float(m.group(1).replace(",", ""))
        req.unit = "jobs" if m.group(2).lower().startswith("job") else "pcs"
    if m := _DEADLINE.search(text):
        n = int(_WORDS.get(m.group(1).lower(), m.group(1)))
        req.deadline_days = n * {"d": 1, "w": 7, "m": 30}[m.group(2)[0].lower()]
    low = text.lower()
    for city in CITY_TO_STATE:
        if re.search(rf"\b{re.escape(city)}\b", low):
            req.delivery_location, req.delivery_state = city.title(), state_for(city)
            break
    return req


class RuleBasedLLM:
    name = "rules"

    async def parse_requirement(self, text: str) -> tuple[SourcingRequirement, LLMResult]:
        return rule_based_parse(text), LLMResult(source="rules")


# ----------------------------------------------------------------------------------------
# Gemini implementation
# ----------------------------------------------------------------------------------------
class GeminiLLM:
    def __init__(self, model: str, client=None, *, api_key: str | None = None, timeout: float = 30.0,
                 retry_attempts: int = 4, company: str = "Rohida Furniture Works", today: date | None = None,
                 rpm: int = 10_000, max_wait_s: float = 90.0):
        if client is None:
            from app.services.embeddings import make_client

            client = make_client(api_key, retry_attempts, timeout)
        self.client, self.model, self.company, self._today = client, model, company, today
        self.name = f"gemini:{model}"
        self.limiter, self.max_wait_s = limiter_for(f"generate:{model}", rpm), max_wait_s

    def today(self) -> date:
        return self._today or date.today()

    async def _generate(self, instructions: str, user: str, fmt: type[BaseModel]):
        from google.genai import types

        t0 = time.perf_counter()
        config = types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_json_schema=fmt.model_json_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # no tools here
        )
        resp = await paced_call(
            lambda: self.client.aio.models.generate_content(model=self.model, contents=user, config=config),
            self.limiter, self.max_wait_s)
        ms = int((time.perf_counter() - t0) * 1000)
        u = getattr(resp, "usage_metadata", None)
        result = LLMResult(
            source=self.name, latency_ms=ms,
            input_tokens=getattr(u, "prompt_token_count", 0) or 0,
            # thinking tokens are billed as output, so count them
            output_tokens=(getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
        )
        text = resp.text
        if not text:  # blocked by safety filters, or empty
            raise ValueError("model returned no text")
        return fmt.model_validate_json(text), result  # Pydantic is the final gatekeeper

    def postprocess(self, parsed: ParsedRequirement, text: str) -> SourcingRequirement:
        """Everything code does to the model's raw output (kept separate so evals can cache raw output)."""
        req = to_requirement(parsed, self.today())
        if req.quantity is not None and req.unit is None:
            # The model found the number but not the unit ("600 rolls"): the rule parser can read the
            # unit off the same text. Only used when both agree on the quantity.
            rules = rule_based_parse(text)
            if rules.quantity == req.quantity and rules.unit:
                req.unit = rules.unit
        return req

    async def parse_requirement(self, text: str) -> tuple[SourcingRequirement, LLMResult]:
        instructions = PARSE_INSTRUCTIONS.format(company=self.company, today=self.today().isoformat())
        try:
            parsed, result = await self._generate(instructions, text, ParsedRequirement)
            result.extra["raw"] = parsed.model_dump()
            return self.postprocess(parsed, text), result
        except Exception as e:  # noqa: BLE001 - any failure falls back, and is recorded
            log.warning("LLM parse failed, using rules: %s", e)
            return rule_based_parse(text), LLMResult(source="rules(fallback)", error=f"{type(e).__name__}: {e}")

def get_llm(settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    if s.gemini_api_key:
        if not s.gemini_model:
            raise RuntimeError("GEMINI_API_KEY is set but GEMINI_MODEL is empty: set GEMINI_MODEL in .env")
        return GeminiLLM(s.gemini_model, api_key=s.gemini_api_key, timeout=s.llm_timeout_s,
                         retry_attempts=s.llm_retry_attempts, company=s.company_name,
                         rpm=s.gemini_rpm, max_wait_s=s.llm_max_rate_wait_s)
    log.warning("GEMINI_API_KEY not set: using the rule-based parser")
    return RuleBasedLLM()
