"""City -> state lookup for delivery locations. Deterministic code, not an LLM call."""

CITY_TO_STATE = {
    "jodhpur": "Rajasthan", "jaipur": "Rajasthan", "udaipur": "Rajasthan", "kishangarh": "Rajasthan",
    "ajmer": "Rajasthan", "bikaner": "Rajasthan", "kota": "Rajasthan",
    "ahmedabad": "Gujarat", "rajkot": "Gujarat", "morbi": "Gujarat", "surat": "Gujarat",
    "delhi": "Delhi", "new delhi": "Delhi", "faridabad": "Haryana", "gurugram": "Haryana",
    "yamunanagar": "Haryana", "noida": "Uttar Pradesh", "saharanpur": "Uttar Pradesh",
    "mumbai": "Maharashtra", "pune": "Maharashtra", "bhiwandi": "Maharashtra",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "chennai": "Tamil Nadu", "kochi": "Kerala",
}


def state_for(city: str | None) -> str | None:
    if not city:
        return None
    return CITY_TO_STATE.get(city.strip().lower())
