"""Fixed vocabularies shared by the LLM schemas, the parser and the vendor data."""

from typing import Literal

CATEGORIES = (
    "Solid timber & lumber",
    "Panel boards",
    "Laminates & veneers",
    "Furniture hardware",
    "Foam & cushioning",
    "Upholstery fabric & leatherette",
    "Adhesives, finishes & coatings",
    "Metal components",
    "Edge banding & profiles",
    "Glass & mirrors",
    "Packaging",
    "Services (job work)",
)
Category = Literal[
    "Solid timber & lumber",
    "Panel boards",
    "Laminates & veneers",
    "Furniture hardware",
    "Foam & cushioning",
    "Upholstery fabric & leatherette",
    "Adhesives, finishes & coatings",
    "Metal components",
    "Edge banding & profiles",
    "Glass & mirrors",
    "Packaging",
    "Services (job work)",
]

UNITS = ("cft", "sheets", "pcs", "kg", "m", "L", "sq ft", "jobs")
Unit = Literal["cft", "sheets", "pcs", "kg", "m", "L", "sq ft", "jobs"]

_UNIT_ALIASES = {
    "sheet": "sheets", "sheets": "sheets", "cft": "cft", "kg": "kg", "kgs": "kg",
    "pc": "pcs", "pcs": "pcs", "piece": "pcs", "pieces": "pcs", "pair": "pcs", "pairs": "pcs",
    "box": "pcs", "boxes": "pcs", "set": "pcs", "sets": "pcs", "nos": "pcs", "units": "pcs",
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m", "mtr": "m",
    "l": "L", "litre": "L", "litres": "L", "liter": "L", "liters": "L",
    "sq ft": "sq ft", "sqft": "sq ft", "roll": "pcs", "rolls": "pcs", "job": "jobs", "jobs": "jobs",
}


def norm_unit(u: str | None) -> str | None:
    return _UNIT_ALIASES.get(u.strip().lower(), u.strip()) if u else None
