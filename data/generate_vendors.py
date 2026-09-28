#!/usr/bin/env python3
"""
Synthetic vendor dataset for Rohida Furniture Works (a FICTIONAL furniture maker in Jodhpur).

Everything here is made up. Contact emails use plus-addressing on a test inbox you control,
so nothing can ever reach a real business.

Outputs (in --out):
  vendors.csv / vendors.json   -> what the app loads into Postgres (no hidden labels)
  vendor_truth.jsonl           -> structured ground truth per vendor, for evaluation ONLY
  hard_cases.csv               -> which vendors were planted as hard cases, and why
  seed_requests.json           -> the 12 seed sourcing requests + gold vendor ids

Usage:
  python generate_vendors.py --n 500 --seed 42 --out data --eval-out eval
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

# --------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------
TIMBER, PANEL, LAM, HW, FOAM, FABRIC, ADH, METAL, EDGE, GLASS, PACK, SERV = (
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
CATEGORY_WEIGHTS = {TIMBER: 15, PANEL: 15, LAM: 8, HW: 12, FOAM: 7, FABRIC: 8,
                    ADH: 8, METAL: 6, EDGE: 4, GLASS: 4, PACK: 8, SERV: 5}
UNIT = {TIMBER: "cft", PANEL: "sheets", LAM: "sheets", HW: "pcs", FOAM: "kg", FABRIC: "m",
        ADH: "L", METAL: "pcs", EDGE: "m", GLASS: "sq ft", PACK: "pcs", SERV: "jobs"}
MOQ_RANGE = {TIMBER: (50, 500), PANEL: (50, 300), LAM: (50, 200), HW: (200, 5000),
             FOAM: (100, 1000), FABRIC: (100, 1000), ADH: (50, 500), METAL: (100, 1000),
             EDGE: (500, 5000), GLASS: (100, 1000), PACK: (500, 5000), SERV: (1, 1)}
LEAD_RANGE = {TIMBER: (7, 30), PANEL: (3, 21), LAM: (3, 15), HW: (3, 20), FOAM: (3, 15),
              FABRIC: (5, 30), ADH: (2, 10), METAL: (10, 35), EDGE: (3, 12), GLASS: (5, 20),
              PACK: (3, 14), SERV: (3, 15)}

# --------------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------------
REGION_WEIGHTS = {"Rajasthan": 35, "Gujarat": 20, "Delhi NCR": 20, "Maharashtra": 10,
                  "UP & Haryana": 10, "South India": 5}
CITIES = {
    "Rajasthan": [("Jodhpur", "Rajasthan", 5), ("Jaipur", "Rajasthan", 3),
                  ("Udaipur", "Rajasthan", 1.5), ("Kishangarh", "Rajasthan", 1)],
    "Gujarat": [("Ahmedabad", "Gujarat", 3), ("Rajkot", "Gujarat", 2), ("Morbi", "Gujarat", 1)],
    "Delhi NCR": [("Delhi", "Delhi", 3), ("Faridabad", "Haryana", 2), ("Noida", "Uttar Pradesh", 2)],
    "Maharashtra": [("Mumbai", "Maharashtra", 2), ("Bhiwandi", "Maharashtra", 1.5),
                    ("Pune", "Maharashtra", 1.5)],
    "UP & Haryana": [("Yamunanagar", "Haryana", 3), ("Saharanpur", "Uttar Pradesh", 2)],
    "South India": [("Bengaluru", "Karnataka", 2), ("Chennai", "Tamil Nadu", 1.5),
                    ("Kochi", "Kerala", 2)],
}
# Categories each region is strong in (weight x3 when picking a region for that category)
STRONG = {
    "Rajasthan": {TIMBER, SERV, ADH, PACK},
    "Gujarat": {HW, LAM, PANEL, GLASS},
    "Delhi NCR": {HW, FABRIC, FOAM, METAL},
    "Maharashtra": {LAM, ADH, HW},
    "UP & Haryana": {PANEL, TIMBER},
    "South India": {TIMBER, FOAM},
}
NEIGHBOURS = {
    "Rajasthan": ["Gujarat", "Madhya Pradesh", "Haryana", "Delhi", "Punjab"],
    "Gujarat": ["Rajasthan", "Maharashtra", "Madhya Pradesh"],
    "Delhi": ["Haryana", "Uttar Pradesh", "Rajasthan"],
    "Haryana": ["Delhi", "Punjab", "Uttar Pradesh", "Rajasthan"],
    "Uttar Pradesh": ["Delhi", "Haryana", "Uttarakhand", "Madhya Pradesh"],
    "Maharashtra": ["Gujarat", "Karnataka", "Madhya Pradesh", "Goa"],
    "Karnataka": ["Tamil Nadu", "Kerala", "Maharashtra", "Andhra Pradesh"],
    "Tamil Nadu": ["Karnataka", "Kerala", "Andhra Pradesh"],
    "Kerala": ["Tamil Nadu", "Karnataka"],
}

# --------------------------------------------------------------------------------------
# Names, contacts, notes
# --------------------------------------------------------------------------------------
PREFIXES = ["Shree Ganesh", "Shri Balaji", "Maruti", "Jai Ambe", "Mahaveer", "Krishna", "Om Sai",
            "Marwar", "Mewar", "Sagar", "Laxmi", "Vinayak", "Kesar", "Aravali", "Thar",
            "Surya", "Chandra", "Navkar", "Sanghvi", "Bhandari", "Rathi", "Agarwal", "Goyal",
            "Mittal", "Patel", "Shah", "Mehta", "Joshi", "Kapoor", "Malhotra", "Nair", "Reddy",
            "Singhal", "Jain", "Oswal", "Kothari", "Bansal", "Vardhman", "Siddhi", "Shubh",
            "Anand", "Prakash", "Royal", "National", "Classic", "Supreme", "Modern", "Star",
            "Unique", "Pioneer", "Heritage", "Sunrise", "Evergreen", "Rajdhani", "Ganga",
            "Narmada", "Kaveri", "Everest", "Kamdhenu", "Padmavati"]
CAT_WORDS = {
    TIMBER: ["Timber Mart", "Saw Mill", "Wood Industries", "Timbers", "Lumber Co."],
    PANEL: ["Plywood House", "Ply & Boards", "Panel Products", "Plywood Industries", "Board Depot"],
    LAM: ["Laminates", "Decor Surfaces", "Veneer House", "Laminate Gallery"],
    HW: ["Hardware", "Hardware Traders", "Fittings", "Hardware Mart"],
    FOAM: ["Foam Industries", "Foams", "Polyfoam", "Cushion Works"],
    FABRIC: ["Fabrics", "Textiles", "Furnishings", "Upholstery House"],
    ADH: ["Coatings", "Paints & Chemicals", "Adhesives", "Polishes"],
    METAL: ["Steel Works", "Metal Fab", "Engineering Works", "Fabricators"],
    EDGE: ["Edge Solutions", "Profiles", "Polymers"],
    GLASS: ["Glass House", "Glass & Mirrors", "Glass Works"],
    PACK: ["Packaging", "Packers", "Corrugators", "Pack Solutions"],
    SERV: ["CNC Works", "Finishing Services", "Powder Coaters", "Seasoning Works"],
}
SUFFIXES = ["", "", "", " & Sons", " Pvt. Ltd.", " Enterprises", " & Co.", " (India)"]
FIRST = ["Ramesh", "Suresh", "Mahesh", "Kavita", "Sunita", "Anil", "Vikram", "Pooja", "Neha",
         "Rajesh", "Deepak", "Manoj", "Priya", "Arjun", "Rohit", "Sanjay", "Meena", "Nitin",
         "Gaurav", "Asha", "Karan", "Farhan", "Imran", "Harpreet", "Lakshmi", "Venkat", "Divya"]
LAST = ["Sharma", "Gehlot", "Rathore", "Choudhary", "Jain", "Agarwal", "Patel", "Shah", "Mehta",
        "Verma", "Gupta", "Singh", "Khan", "Kumar", "Nair", "Iyer", "Reddy", "Bishnoi", "Soni",
        "Mathur", "Bhati", "Purohit", "Kothari", "Malik"]
RATING_NOTES = [
    "Reliable on quality, slow to reply on WhatsApp.",
    "Fast dispatch; prices slightly above market.",
    "Good rates for bulk orders; quality varies between batches.",
    "Long-standing supplier, easy to negotiate with.",
    "New vendor, no order history yet.",
    "Replies quickly, but delivery dates often slip by a few days.",
    "Consistent quality; insists on full advance for first order.",
    "Helpful with samples; limited stock of less common sizes.",
    "Occasional short supply in peak wedding season.",
    "Good packaging and transit care; small team, one contact person.",
    "Cheap, but needs close quality checks on arrival.",
    "Premium supplier, used for hotel projects.",
]
PAYMENT_TERMS = [
    "100% advance", "50% advance, balance before dispatch",
    "30% advance, balance before dispatch", "Payment against delivery",
    "15 days credit after first order", "30 days credit for regular customers",
]
EXTRA_LINES = [
    "Custom sizes on request.", "Dispatch from our {city} godown.",
    "Regular supplier to furniture manufacturers in {state}.",
    "Samples available before bulk order.", "Transport arranged at actual cost.",
    "Stock updated weekly; call to confirm availability.", "", "", "",
]

# --------------------------------------------------------------------------------------
# Product generators. Each returns {"product", "attrs", "price": (lo, hi), "price_unit"}.
# --------------------------------------------------------------------------------------
TIMBER_PRICE = {"sheesham": (2200, 3200), "mango": (1100, 1600), "acacia": (1300, 1900),
                "teak": (3800, 6500), "rubberwood": (1400, 2000)}
TIMBER_NAME = {"sheesham": "Sheesham", "mango": "Mango wood", "acacia": "Acacia",
               "teak": "Teak", "rubberwood": "Rubberwood"}
TIMBER_SYN = {"sheesham": ["Indian rosewood", "Dalbergia sissoo", "Shisham"],
              "mango": ["Mangifera indica (aam) wood"], "teak": ["Tectona grandis", "Sagwan"],
              "rubberwood": ["Hevea wood"], "acacia": ["Acacia"]}


def p_timber(rng, species=None, seasoning=None, moisture=None, thickness=None, region=None):
    if species is None:
        if region == "South India":
            species = rng.choices(["rubberwood", "mango", "teak"], [70, 15, 15])[0]
        else:
            species = rng.choices(list(TIMBER_PRICE), [35, 25, 15, 12, 3])[0]
    if seasoning is None:
        # Planted scenario for seed request 11: acacia is only ever sold air-dried.
        seasoning = "air-dried" if species == "acacia" else rng.choices(
            ["kiln-dried", "air-dried"], [85, 15])[0]
    if seasoning == "kiln-dried":
        moisture = moisture or rng.choice([(8, 10), (10, 12), (10, 12), (12, 14)])
    else:
        moisture = None
    thickness = thickness or sorted(rng.sample([1, 1.25, 1.5, 2, 2.5, 3], rng.randint(2, 4)))
    length = rng.choice([(4, 8), (6, 10), (6, 12), (8, 14)])
    return {"product": "timber", "attrs": {
        "species": species, "seasoning": seasoning,
        "moisture_pct": list(moisture) if moisture else None,
        "thickness_in": thickness, "length_ft": list(length)},
        "price": TIMBER_PRICE[species], "price_unit": "cft"}


PANEL_BOARDS = {  # key: (label, synonym label, price per 8x4 sheet, thickness options mm)
    "plywood_mr": ("Moisture-resistant plywood", "Moisture-resistant ply", (1800, 2600), [6, 9, 12, 16, 18]),
    "plywood_wp": ("Waterproof plywood", "Waterproof ply for wet areas", (2800, 4200), [6, 9, 12, 16, 18, 25]),
    "mdf": ("Plain MDF board", "Medium density fibreboard", (1100, 1600), [6, 9, 12, 16, 18, 25]),
    "prelam_mdf": ("Pre-laminated MDF", "Pre-lam medium density fibreboard", (1600, 2400), [9, 12, 16, 18, 25]),
    "hdhmr": ("HDHMR board", "High-density moisture-resistant board", (2200, 3000), [9, 12, 16, 18]),
    "particle": ("Plain particle board", "Chipboard", (800, 1200), [9, 12, 16, 18, 25]),
    "prelam_particle": ("Pre-laminated particle board", "Pre-laminated chipboard", (1100, 1600), [12, 16, 18, 25]),
    "blockboard": ("Block board", "Block board (wooden core)", (2000, 2800), [19, 25]),
}
FINISHES = ["walnut", "oak", "white", "teak", "wenge", "grey", "maple"]


def p_panel(rng, kind=None, thickness=None, finishes=None, **_):
    kind = kind or rng.choices(list(PANEL_BOARDS), [20, 14, 12, 12, 8, 10, 10, 6])[0]
    _, _, price, thks = PANEL_BOARDS[kind]
    thickness = thickness or sorted(rng.sample(thks, rng.randint(min(2, len(thks)), len(thks))))
    attrs = {"board": kind, "thickness_mm": thickness, "sheet_size": "8x4 ft"}
    if kind.startswith("prelam"):
        attrs["finishes"] = finishes or sorted(rng.sample(FINISHES, rng.randint(2, 4)))
    return {"product": "panel", "attrs": attrs, "price": price, "price_unit": "sheet"}


def p_lam(rng, kind=None, **_):
    kind = kind or rng.choices(["laminate_1mm", "liner", "veneer"], [55, 20, 25])[0]
    if kind == "laminate_1mm":
        attrs = {"type": kind, "thickness_mm": 1.0,
                 "textures": sorted(rng.sample(["suede", "gloss", "matte", "high-gloss", "woodgrain", "stone texture"], 3))}
        price = (900, 2200)
    elif kind == "liner":
        attrs = {"type": kind, "thickness_mm": 0.8}
        price = (350, 600)
    else:
        attrs = {"type": kind, "species": sorted(rng.sample(["teak", "walnut", "oak", "ash", "sapele"], 3))}
        price = (1800, 4500)
    return {"product": "laminate", "attrs": attrs, "price": price, "price_unit": "sheet"}


CHANNEL_LENGTHS = [300, 350, 400, 450, 500, 550, 600]


def p_hw(rng, kind=None, soft_close=None, lengths=None, **_):
    kind = kind or rng.choices(["hinge", "drawer_channel", "tandem", "handle", "minifix", "gas_lift"],
                               [22, 25, 10, 20, 10, 13])[0]
    if kind == "hinge":
        sc = rng.random() < 0.7 if soft_close is None else soft_close
        return {"product": "hardware", "attrs": {"item": kind, "soft_close": sc, "cup_mm": 35,
                "overlay": sorted(rng.sample(["full", "half", "inset"], rng.randint(1, 3)))},
                "price": (45, 180), "price_unit": "pc"}
    if kind in ("drawer_channel", "tandem"):
        sc = True if kind == "tandem" else (rng.random() < 0.6 if soft_close is None else soft_close)
        if lengths is None:
            start = rng.randint(0, 3)
            lengths = CHANNEL_LENGTHS[start:start + rng.randint(3, 7)]
        return {"product": "hardware", "attrs": {"item": kind, "soft_close": sc, "lengths_mm": lengths,
                "load_kg": rng.choice([35, 45])},
                "price": (180, 650) if kind == "drawer_channel" else (1200, 2600), "price_unit": "pair"}
    if kind == "handle":
        return {"product": "hardware", "attrs": {"item": kind,
                "material": rng.choice(["stainless steel", "aluminium", "brass", "zinc alloy"]),
                "finish": rng.choice(["matte black", "satin nickel", "antique brass", "chrome", "rose gold"])},
                "price": (40, 400), "price_unit": "pc"}
    if kind == "minifix":
        return {"product": "hardware", "attrs": {"item": kind}, "price": (6, 15), "price_unit": "set"}
    return {"product": "hardware", "attrs": {"item": "gas_lift", "use": rng.choice(["bed", "chair"]),
            "force_n": rng.choice([600, 800, 1000, 1200])}, "price": (350, 1400), "price_unit": "pc"}


def p_foam(rng, kind=None, densities=None, **_):
    kind = kind or rng.choices(["pu_foam", "hr_foam", "memory_foam", "fibre_fill"], [50, 20, 15, 15])[0]
    if kind == "pu_foam":
        d = densities or sorted(rng.sample([23, 28, 32, 40, 50], rng.randint(2, 4)))
        return {"product": "foam", "attrs": {"type": kind, "densities": d}, "price": (160, 320), "price_unit": "kg"}
    if kind == "hr_foam":
        d = densities or sorted(rng.sample([40, 50], rng.randint(1, 2)))
        return {"product": "foam", "attrs": {"type": kind, "densities": d}, "price": (280, 420), "price_unit": "kg"}
    if kind == "memory_foam":
        return {"product": "foam", "attrs": {"type": kind}, "price": (450, 650), "price_unit": "kg"}
    return {"product": "foam", "attrs": {"type": kind}, "price": (140, 220), "price_unit": "kg"}


def p_fabric(rng, kind=None, **_):
    kind = kind or rng.choices(["velvet", "linen_blend", "chenille", "leatherette", "fr_fabric", "stain_resistant"],
                               [20, 18, 15, 20, 17, 10])[0]
    attrs = {"type": kind, "flame_retardant": kind == "fr_fabric", "width_in": rng.choice([54, 56])}
    if kind == "leatherette":
        attrs["base"] = rng.choice(["PU", "PVC"])
    if kind == "fr_fabric":
        attrs["fr_type"] = rng.choice(["inherent", "treated"])
    price = {"velvet": (350, 750), "linen_blend": (300, 650), "chenille": (280, 600),
             "leatherette": (180, 450), "fr_fabric": (450, 950), "stain_resistant": (320, 700)}[kind]
    return {"product": "fabric", "attrs": attrs, "price": price, "price_unit": "m"}


def p_adh(rng, kind=None, **_):
    kind = kind or rng.choice(["pva_glue", "contact_adhesive", "pu_lacquer", "melamine_polish", "wood_stain", "sealer"])
    attrs = {"type": kind}
    if kind == "pu_lacquer":
        attrs["sheen"] = rng.choice(["matte", "satin", "gloss"])
    if kind == "wood_stain":
        attrs["shades"] = sorted(rng.sample(["walnut", "teak", "mahogany", "ebony", "natural"], 3))
    price = {"pva_glue": (120, 220), "contact_adhesive": (250, 450), "pu_lacquer": (380, 650),
             "melamine_polish": (220, 400), "wood_stain": (300, 550), "sealer": (200, 380)}[kind]
    return {"product": "finish", "attrs": attrs, "price": price, "price_unit": "L"}


def p_metal(rng, kind=None, **_):
    kind = kind or rng.choice(["ms_legs", "ss_legs", "bed_frame", "powder_coated_base", "chair_frame"])
    price = {"ms_legs": (180, 450), "ss_legs": (450, 1200), "bed_frame": (3500, 8000),
             "powder_coated_base": (900, 2500), "chair_frame": (800, 2200)}[kind]
    return {"product": "metal", "attrs": {"type": kind, "custom_drawings": rng.random() < 0.6,
            "finish": rng.choice(["powder coated black", "powder coated white", "brushed", "gold PVD", "raw"])},
            "price": price, "price_unit": "pc"}


def p_edge(rng, kind=None, **_):
    kind = kind or rng.choices(["pvc_edge", "abs_edge", "alu_profile"], [50, 30, 20])[0]
    if kind == "alu_profile":
        return {"product": "edge", "attrs": {"type": kind, "finish": rng.choice(["anodised silver", "black", "champagne"])},
                "price": (60, 180), "price_unit": "m"}
    return {"product": "edge", "attrs": {"type": kind,
            "thickness_mm": sorted(rng.sample([0.8, 2.0], rng.randint(1, 2))),
            "width_mm": sorted(rng.sample([22, 45], rng.randint(1, 2)))},
            "price": (4, 14) if kind == "pvc_edge" else (6, 18), "price_unit": "m"}


def p_glass(rng, kind=None, **_):
    kind = kind or rng.choice(["toughened", "back_painted", "mirror"])
    thk = {"toughened": [8, 10, 12], "back_painted": [4, 5, 6], "mirror": [4, 5]}[kind]
    price = {"toughened": (140, 260), "back_painted": (180, 320), "mirror": (80, 160)}[kind]
    return {"product": "glass", "attrs": {"type": kind, "thickness_mm": sorted(rng.sample(thk, rng.randint(1, len(thk))))},
            "price": price, "price_unit": "sq ft"}


def p_pack(rng, kind=None, ply=None, custom=None, **_):
    kind = kind or rng.choices(["box", "epe_sheet", "corner_protector", "stretch_wrap"], [45, 20, 20, 15])[0]
    if kind == "box":
        ply = ply or sorted(rng.sample([3, 5, 7], rng.randint(1, 3)))
        custom = rng.random() < 0.75 if custom is None else custom
        return {"product": "packaging", "attrs": {"type": kind, "ply": ply, "custom_size": custom},
                "price": (35, 180), "price_unit": "pc"}
    if kind == "epe_sheet":
        return {"product": "packaging", "attrs": {"type": kind, "thickness_mm": sorted(rng.sample([2, 5, 10, 20], 2))},
                "price": (8, 60), "price_unit": "pc"}
    if kind == "corner_protector":
        return {"product": "packaging", "attrs": {"type": kind, "material": rng.choice(["EPE", "paper", "plastic"])},
                "price": (2, 9), "price_unit": "pc"}
    return {"product": "packaging", "attrs": {"type": kind, "micron": rng.choice([20, 23, 25])},
            "price": (160, 260), "price_unit": "roll"}


def p_serv(rng, kind=None, **_):
    kind = kind or rng.choice(["cnc_routing", "powder_coating", "polishing", "kiln_drying"])
    price = {"cnc_routing": (25, 60), "powder_coating": (18, 40), "polishing": (35, 90), "kiln_drying": (120, 250)}[kind]
    unit = {"cnc_routing": "sq ft", "powder_coating": "sq ft", "polishing": "sq ft", "kiln_drying": "cft"}[kind]
    return {"product": "service", "attrs": {"type": kind}, "price": price, "price_unit": unit}


GEN = {TIMBER: p_timber, PANEL: p_panel, LAM: p_lam, HW: p_hw, FOAM: p_foam, FABRIC: p_fabric,
       ADH: p_adh, METAL: p_metal, EDGE: p_edge, GLASS: p_glass, PACK: p_pack, SERV: p_serv}


# --------------------------------------------------------------------------------------
# Free-text descriptions (varied phrasing so vector search has real work to do)
# --------------------------------------------------------------------------------------
def fmt_list(xs, suffix=""):
    xs = [f"{x:g}" if isinstance(x, float) else str(x) for x in xs]
    body = xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    return body + suffix


def describe(rng, p, synonym=False):
    a = p["attrs"]
    kind = p["product"]
    if kind == "timber":
        name = rng.choice(TIMBER_SYN[a["species"]]) if synonym else TIMBER_NAME[a["species"]]
        thk = fmt_list(a["thickness_in"], " inch")
        l1, l2 = a["length_ft"]
        if a["seasoning"] == "kiln-dried":
            m1, m2 = a["moisture_pct"]
            return rng.choice([
                f"Kiln-dried {name} planks, {thk} thickness, lengths {l1}–{l2} ft, moisture content {m1}–{m2}%.",
                f"We supply {name} timber in {thk} sections, kiln seasoned to {m1}-{m2}% moisture, random lengths {l1}-{l2} ft.",
                f"{name} sawn planks ({thk}), chamber dried, moisture {m1} to {m2} percent, {l1} to {l2} feet long.",
            ])
        return rng.choice([
            f"Air-dried {name} planks, {thk} thickness, lengths {l1}–{l2} ft. Natural seasoning, moisture not tested.",
            f"{name} timber, open-yard seasoned, {thk} sections, {l1}-{l2} ft lengths.",
        ])
    if kind == "panel":
        label, syn, _, _ = PANEL_BOARDS[a["board"]]
        name = syn if synonym else label
        s = f"{name}, {fmt_list(a['thickness_mm'], ' mm')}, 8x4 ft sheets."
        if "finishes" in a:
            s += f" Finishes: {fmt_list(a['finishes'])}."
        if a["board"] == "plywood_wp":
            s += rng.choice([" Suitable for kitchens and bathrooms.", " Safe for wet areas and outdoor furniture.",""])
        if a["board"] == "plywood_mr":
            s += rng.choice([" For dry interior use only.", " Interior use.", ""])
        return s
    if kind == "laminate":
        if a["type"] == "laminate_1mm":
            return f"1 mm decorative laminates in {fmt_list(a['textures'])} finishes, 8x4 ft."
        if a["type"] == "liner":
            return "0.8 mm liner / balancing laminate for the inside of shutters and carcasses."
        return f"Natural wood veneers: {fmt_list(a['species'])}, 0.5 mm face, 8x4 ft."
    if kind == "hardware":
        it = a["item"]
        if it == "hinge":
            return f"{'Soft-close' if a['soft_close'] else 'Normal'} concealed hinges, 35 mm cup, {fmt_list(a['overlay'])} overlay."
        if it == "drawer_channel":
            sc = rng.choice(["soft-close", "soft closing", "self-closing damped"]) if a["soft_close"] else "regular (non soft-close)"
            return (f"Telescopic ball-bearing drawer channels, {sc}, lengths {fmt_list(a['lengths_mm'], ' mm')}, "
                    f"load {a['load_kg']} kg, sold in pairs.")
        if it == "tandem":
            return f"Tandem box drawer systems with soft-close runners, {fmt_list(a['lengths_mm'], ' mm')}, load {a['load_kg']} kg."
        if it == "handle":
            return f"Cabinet and wardrobe handles in {a['material']}, {a['finish']} finish, 96–320 mm centres."
        if it == "minifix":
            return "Minifix cam-lock connectors and dowels for knock-down furniture."
        return f"Gas lifts for {a['use']}s, {a['force_n']} N."
    if kind == "foam":
        t = a["type"]
        if t == "pu_foam":
            return rng.choice([
                f"PU foam sheets and blocks in {fmt_list(a['densities'])} density (kg/m³), cut to size.",
                f"Polyurethane foam, densities {fmt_list(a['densities'])}, for sofa seats, backs and mattresses.",
            ])
        if t == "hr_foam":
            return f"HR (high-resilience) foam, {fmt_list(a['densities'])} density, for premium seating."
        if t == "memory_foam":
            return "Memory foam sheets for mattresses and cushion toppers."
        return "Hollow conjugated fibre fill for cushions and pillows."
    if kind == "fabric":
        t = a["type"]
        w = a["width_in"]
        if t == "fr_fabric":
            fr = "inherently flame-retardant polyester" if a["fr_type"] == "inherent" else "flame-retardant treated upholstery fabric"
            return f"{fr.capitalize()} for hotels and hospitality projects, {w} inch width, 40+ colours."
        if t == "stain_resistant":
            return f"Stain-resistant, easy-clean upholstery fabric, {w} inch width. Ideal for dining chairs and hotel lobbies."
        if t == "leatherette":
            return f"{a['base']} leatherette for sofas and headboards, {w} inch width."
        return {"velvet": f"Upholstery velvet, {w} inch width, plain and embossed.",
                "linen_blend": f"Linen-look blended upholstery fabric, {w} inch width.",
                "chenille": f"Chenille sofa fabric, {w} inch width, jacquard designs."}[t]
    if kind == "finish":
        t = a["type"]
        return {"pva_glue": "PVA wood adhesive for joinery and lamination.",
                "contact_adhesive": "Synthetic rubber contact adhesive for laminates and foam.",
                "pu_lacquer": f"PU lacquer, {a.get('sheen', 'matte')} finish, with sealer and thinner.",
                "melamine_polish": "Melamine polish system for solid wood furniture.",
                "wood_stain": f"Wood stains in {fmt_list(a.get('shades', ['natural']))} shades.",
                "sealer": "Sanding sealer for wood and MDF."}[t]
    if kind == "metal":
        nice = a["type"].replace("_", " ").replace("ms", "MS").replace("ss", "SS")
        s = f"{nice.capitalize()} for furniture, {a['finish']}."
        return s + (" Made to your drawings." if a["custom_drawings"] else " Catalogue designs only.")
    if kind == "edge":
        if a["type"] == "alu_profile":
            return f"Aluminium profiles for shutters and glass frames, {a['finish']}."
        mat = "PVC" if a["type"] == "pvc_edge" else "ABS"
        return f"{mat} edge banding tape, {fmt_list(a['thickness_mm'], ' mm')} thick, {fmt_list(a['width_mm'], ' mm')} wide, colour-matched to laminates."
    if kind == "glass":
        t = a["type"]
        name = {"toughened": "Toughened glass tops", "back_painted": "Back-painted glass", "mirror": "Mirrors"}[t]
        return f"{name}, {fmt_list(a['thickness_mm'], ' mm')}, cut and edge-polished to size."
    if kind == "packaging":
        t = a["type"]
        if t == "box":
            s = f"{fmt_list(a['ply'], '-ply')} corrugated boxes"
            return s + (", custom sizes printed or plain." if a["custom_size"] else ", fixed catalogue sizes only.")
        if t == "epe_sheet":
            return f"EPE foam sheets {fmt_list(a['thickness_mm'], ' mm')} for wrapping furniture."
        if t == "corner_protector":
            return f"{a['material']} corner and edge protectors for furniture packing."
        return f"Stretch wrap film rolls, {a['micron']} micron."
    if kind == "service":
        return {"cnc_routing": "CNC routing and nesting for panels and carved panels, job work.",
                "powder_coating": "Powder coating job work for furniture frames and legs.",
                "polishing": "Furniture polishing job work: PU, melamine and duco.",
                "kiln_drying": "Timber seasoning / kiln drying job work for your own stock."}[a["type"]]
    return a.get("description", "")


# --------------------------------------------------------------------------------------
# Vendor assembly
# --------------------------------------------------------------------------------------
def nice_round(x):
    if x <= 20:
        return max(1, int(round(x)))
    step = 10 if x < 200 else 50 if x < 1000 else 100 if x < 5000 else 500
    return int(round(x / step) * step)


def inr(x):
    return f"₹{x:,.0f}" if x >= 100 else f"₹{x:,.1f}".replace(".0", "")


class Factory:
    def __init__(self, seed, email_base):
        self.rng = random.Random(seed)
        self.local, self.domain = email_base.split("@")
        self.used_names = set()
        self.counter = 0

    def new_id(self):
        self.counter += 1
        return f"V-{self.counter:04d}"

    def pick_region(self, category):
        regs = list(REGION_WEIGHTS)
        w = [REGION_WEIGHTS[r] * (3 if category in STRONG[r] else 1) for r in regs]
        return self.rng.choices(regs, w)[0]

    def pick_city(self, region):
        opts = CITIES[region]
        c = self.rng.choices(opts, [o[2] for o in opts])[0]
        return c[0], c[1]

    def name(self, category):
        for _ in range(200):
            n = f"{self.rng.choice(PREFIXES)} {self.rng.choice(CAT_WORDS[category])}{self.rng.choice(SUFFIXES)}"
            if n not in self.used_names:
                self.used_names.add(n)
                return n
        raise RuntimeError("ran out of unique names")

    def served(self, state):
        if self.rng.random() < 0.2:
            return ["Pan-India"]
        nb = NEIGHBOURS.get(state, [])
        return [state] + self.rng.sample(nb, self.rng.randint(0, len(nb)))

    def build(self, category, products=None, state_region=None, text_override=None,
              synonym=False, moq=None, lead=None, regions=None):
        rng = self.rng
        vid = self.new_id()
        region = state_region or self.pick_region(category)
        city, state = self.pick_city(region)
        q = rng.random()  # where this vendor sits in the price range
        if products is None:
            n = 1 if category == SERV else rng.choices([1, 2, 3], [45, 35, 20])[0]
            products, seen = [], set()
            for _ in range(n * 3):
                p = GEN[category](rng, region=region)
                key = json.dumps(p["attrs"].get("board") or p["attrs"].get("item") or p["attrs"].get("type")
                                 or p["attrs"].get("species"))
                if key not in seen:
                    seen.add(key)
                    products.append(p)
                if len(products) == n:
                    break
        categories = [category]
        # ~10% of panel-board dealers also stock laminates (same unit: sheets)
        if category == PANEL and text_override is None and rng.random() < 0.25:
            products.append(p_lam(rng))
            categories.append(LAM)

        if text_override is not None:
            specs = text_override
        else:
            specs = " ".join(describe(rng, p, synonym=synonym) for p in products)
            extra = rng.choice(EXTRA_LINES).format(city=city, state=state)
            specs = (specs + " " + extra).strip()

        lo, hi = MOQ_RANGE[category]
        moq_v = moq if moq is not None else nice_round(lo + (hi - lo) * rng.random() ** 1.5)
        l1, l2 = LEAD_RANGE[category]
        lead_v = lead if lead is not None else rng.randint(l1, l2) + (0 if state == "Rajasthan" else rng.randint(2, 6))
        if any(p["attrs"].get("board") == "plywood_wp" for p in products) and lead is None:
            lead_v = max(lead_v, 7)  # planted scenario for seed request 12

        plo, phi = products[0]["price"]
        mid = plo + (phi - plo) * q
        band = "budget" if q < 0.33 else "mid" if q < 0.7 else "premium"
        price_txt = f"{inr(nice_round(mid * 0.94) if mid > 20 else mid * 0.94)}–{inr(nice_round(mid * 1.06) if mid > 20 else mid * 1.06)} per {products[0]['price_unit']}"

        cname = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        vendor = {
            "vendor_id": vid,
            "name": self.name(category),
            "city": city,
            "state": state,
            "regions_served": regions if regions is not None else self.served(state),
            "categories": categories,
            "specs": specs,
            "moq": moq_v,
            "moq_unit": UNIT[category],
            "lead_time_days": lead_v,
            "price_band": band,
            "indicative_price": price_txt,
            "payment_terms": rng.choice(PAYMENT_TERMS),
            "gst_registered": rng.random() < 0.95,
            "rating_notes": rng.choice(RATING_NOTES),
            "contact_name": cname,
            "contact_email": f"{self.local}+{vid.lower().replace('-', '')}@{self.domain}",
            "phone": f"+91-00000-{int(vid[2:]):05d}",  # deliberately invalid number
        }
        truth = {"vendor_id": vid, "products": [{"product": p["product"], **p["attrs"]} for p in products],
                 "hard_case": None, "target_request": None}
        return vendor, truth


# --------------------------------------------------------------------------------------
# Seed requests and gold matching
# --------------------------------------------------------------------------------------
def serves_raj(v):
    return "Pan-India" in v["regions_served"] or "Rajasthan" in v["regions_served"]


SEED_REQUESTS = [
    {"id": "R01", "kind": "clear", "qty": 300, "deadline_days": 21,
     "text": "300 cft of kiln-dried sheesham, 1.5 inch thick, moisture ≤12%, delivered to Jodhpur within 3 weeks.",
     "match": lambda p: p["product"] == "timber" and p["species"] == "sheesham" and p["seasoning"] == "kiln-dried"
     and p["moisture_pct"][1] <= 12 and 1.5 in p["thickness_in"]},
    {"id": "R02", "kind": "clear", "qty": 200, "deadline_days": 20,
     "text": "200 sheets of 18 mm waterproof plywood for a hotel bathroom vanity order in Udaipur, within 20 days.",
     "match": lambda p: p["product"] == "panel" and p["board"] == "plywood_wp" and 18 in p["thickness_mm"]},
    {"id": "R03", "kind": "clear", "qty": 1500, "deadline_days": 14,
     "text": "1,500 pairs of 450 mm soft-close telescopic drawer channels, delivered to Jodhpur within 2 weeks.",
     "match": lambda p: p["product"] == "hardware" and p["item"] == "drawer_channel" and p["soft_close"]
     and 450 in p["lengths_mm"]},
    {"id": "R04", "kind": "clear", "qty": 800, "deadline_days": 10,
     "text": "800 kg of 40-density PU foam sheets for sofa seats, delivered to Jodhpur within 10 days.",
     "match": lambda p: p["product"] == "foam" and p["type"] == "pu_foam" and 40 in p["densities"]},
    {"id": "R05", "kind": "clear", "qty": 600, "deadline_days": 28,
     "text": "600 m of flame-retardant upholstery fabric for a 120-room hotel project in Jaipur, within 4 weeks.",
     "match": lambda p: p["product"] == "fabric" and p["flame_retardant"]},
    {"id": "R06", "kind": "clear", "qty": 150, "deadline_days": None,
     "text": "150 sheets of 18 mm pre-laminated MDF, walnut finish, delivered to Jodhpur.",
     "match": lambda p: p["product"] == "panel" and p["board"] == "prelam_mdf" and 18 in p["thickness_mm"]
     and "walnut" in p.get("finishes", [])},
    {"id": "R07", "kind": "clear", "qty": 2000, "deadline_days": 7,
     "text": "2,000 five-ply corrugated boxes sized for a dining chair, with EPE corners, delivered to Jodhpur within 1 week.",
     "match": lambda p: p["product"] == "packaging" and p["type"] == "box" and 5 in p["ply"] and p["custom_size"]},
    {"id": "R08", "kind": "vague", "text": "Need some good wood for the new bed range.",
     "expected": "clarify: species, quantity, thickness, deadline"},
    {"id": "R09", "kind": "vague", "text": "Hinges for wardrobes, urgent.",
     "expected": "clarify: quantity, soft-close or not, overlay type, exact deadline"},
    {"id": "R10", "kind": "vague", "text": "Fabric for the hotel sofas.",
     "expected": "clarify: fabric type, flame-retardant needed?, quantity, deadline"},
    {"id": "R11", "kind": "tricky", "qty": 100, "deadline_days": 30,
     "text": "100 cft of kiln-dried acacia, delivered to Jodhpur within a month.",
     "expected": "no vendor fully matches: all acacia stock is air-dried; report the near misses",
     "match": lambda p: p["product"] == "timber" and p["species"] == "acacia" and p["seasoning"] == "kiln-dried"},
    {"id": "R12", "kind": "tricky", "qty": 50, "deadline_days": 5,
     "text": "50 sheets of waterproof plywood, any thickness, delivered to Jodhpur in 5 days.",
     "expected": "no vendor meets the deadline: every waterproof plywood vendor needs 7+ days",
     "match": lambda p: p["product"] == "panel" and p["board"] == "plywood_wp"},
]
# Week 3: 28 more requests, for a 40-request evaluation set (26 clear, 8 vague, 6 tricky).
_hw = lambda p, item: p["product"] == "hardware" and p["item"] == item  # noqa: E731
SEED_REQUESTS += [
    {"id": "R13", "kind": "clear", "qty": 2000, "deadline_days": 10,
     "text": "2,000 soft-close concealed hinges, 35 mm cup, full overlay, delivered to Jodhpur within 10 days.",
     "match": lambda p: _hw(p, "hinge") and p["soft_close"] and "full" in p["overlay"]},
    {"id": "R14", "kind": "clear", "qty": 300, "deadline_days": 15,
     "text": "300 gas lifts for hydraulic storage beds, 1000 N, delivered to Jodhpur within 15 days.",
     "match": lambda p: _hw(p, "gas_lift") and p["use"] == "bed" and p["force_n"] == 1000},
    {"id": "R15", "kind": "clear", "qty": 120, "deadline_days": 12,
     "text": "120 sheets of 1 mm suede-finish decorative laminate, delivered to Jaipur within 12 days.",
     "match": lambda p: p["product"] == "laminate" and p["type"] == "laminate_1mm" and "suede" in p["textures"]},
    {"id": "R16", "kind": "clear", "qty": 80, "deadline_days": 15,
     "text": "80 sheets of natural teak veneer, delivered to Jodhpur within 15 days.",
     "match": lambda p: p["product"] == "laminate" and p["type"] == "veneer" and "teak" in p["species"]},
    {"id": "R17", "kind": "clear", "qty": 500, "deadline_days": 12,
     "text": "500 kg of 50-density HR foam for premium sofas, delivered to Jodhpur within 12 days.",
     "match": lambda p: p["product"] == "foam" and p["type"] == "hr_foam" and 50 in p["densities"]},
    {"id": "R18", "kind": "clear", "qty": 400, "deadline_days": 21,
     "text": "400 m of PU leatherette for bed headboards, delivered to Jodhpur within 3 weeks.",
     "match": lambda p: p["product"] == "fabric" and p["type"] == "leatherette" and p.get("base") == "PU"},
    {"id": "R19", "kind": "clear", "qty": 200, "deadline_days": 7,
     "text": "200 litres of matte PU lacquer with sealer, delivered to Jodhpur within 7 days.",
     "match": lambda p: p["product"] == "finish" and p["type"] == "pu_lacquer" and p.get("sheen") == "matte"},
    {"id": "R20", "kind": "clear", "qty": 300, "deadline_days": 5,
     "text": "300 litres of PVA wood glue for joinery, delivered to Jodhpur within 5 days.",
     "match": lambda p: p["product"] == "finish" and p["type"] == "pva_glue"},
    {"id": "R21", "kind": "clear", "qty": 400, "deadline_days": 30,
     "text": "400 SS legs for dining tables, made to our drawings, delivered to Jodhpur within 30 days.",
     "match": lambda p: p["product"] == "metal" and p["type"] == "ss_legs" and p["custom_drawings"]},
    {"id": "R22", "kind": "clear", "qty": 5000, "deadline_days": 10,
     "text": "5,000 m of 2 mm PVC edge banding, 22 mm wide, delivered to Jodhpur within 10 days.",
     "match": lambda p: p["product"] == "edge" and p["type"] == "pvc_edge" and 2.0 in p["thickness_mm"]
     and 22 in p["width_mm"]},
    {"id": "R23", "kind": "clear", "qty": 600, "deadline_days": 20,
     "text": "600 sq ft of 10 mm toughened glass for dining table tops, delivered to Udaipur within 20 days.",
     "match": lambda p: p["product"] == "glass" and p["type"] == "toughened" and 10 in p["thickness_mm"]},
    {"id": "R24", "kind": "clear", "qty": 300, "deadline_days": 14,
     "text": "300 sq ft of 5 mm mirror for wardrobe shutters, delivered to Jodhpur within 2 weeks.",
     "match": lambda p: p["product"] == "glass" and p["type"] == "mirror" and 5 in p["thickness_mm"]},
    {"id": "R25", "kind": "clear", "qty": 1, "deadline_days": 10,
     "text": "1 job of CNC routing to nest and cut wardrobe panels, in Jodhpur, within 10 days.",
     "match": lambda p: p["product"] == "service" and p["type"] == "cnc_routing"},
    {"id": "R26", "kind": "clear", "qty": 150, "deadline_days": 30,
     "text": "150 cft of kiln-dried teak, 2 inch thick, delivered to Jaipur within 30 days.",
     "match": lambda p: p["product"] == "timber" and p["species"] == "teak" and p["seasoning"] == "kiln-dried"
     and 2 in p["thickness_in"]},
    {"id": "R27", "kind": "clear", "qty": 400, "deadline_days": 25,
     "text": "400 cft of kiln-dried mango wood, 1 inch thick, delivered to Jodhpur within 25 days.",
     "match": lambda p: p["product"] == "timber" and p["species"] == "mango" and p["seasoning"] == "kiln-dried"
     and 1 in p["thickness_in"]},
    {"id": "R28", "kind": "clear", "qty": 100, "deadline_days": 15,
     "text": "100 sheets of 18 mm HDHMR board for kitchen cabinets, delivered to Jodhpur within 15 days.",
     "match": lambda p: p["product"] == "panel" and p["board"] == "hdhmr" and 18 in p["thickness_mm"]},
    {"id": "R29", "kind": "clear", "qty": 600, "deadline_days": 7,
     "text": "600 rolls of 23 micron stretch wrap film, delivered to Jodhpur within 7 days.",
     "match": lambda p: p["product"] == "packaging" and p["type"] == "stretch_wrap" and p["micron"] == 23},
    {"id": "R30", "kind": "clear", "qty": 300, "deadline_days": 14,
     "text": "300 kg of memory foam sheets for mattress toppers, delivered to Jodhpur within 2 weeks.",
     "match": lambda p: p["product"] == "foam" and p["type"] == "memory_foam"},
    {"id": "R31", "kind": "clear", "qty": 800, "deadline_days": 20,
     "text": "800 sets of 500 mm soft-close tandem drawer systems, delivered to Jodhpur within 20 days.",
     "match": lambda p: _hw(p, "tandem") and 500 in p["lengths_mm"]},
    {"id": "R32", "kind": "vague", "text": "Some foam for cushions.", "expected": "clarify: type, density, quantity"},
    {"id": "R33", "kind": "vague", "text": "Need glass for the new tables, ASAP.",
     "expected": "clarify: type, thickness, quantity, exact deadline"},
    {"id": "R34", "kind": "vague", "text": "Laminates for the wardrobe project.",
     "expected": "clarify: finish, quantity, deadline"},
    {"id": "R35", "kind": "vague", "text": "Polish for the dining sets.", "expected": "clarify: type, quantity"},
    {"id": "R36", "kind": "vague", "text": "Boxes for dispatch.", "expected": "clarify: ply, size, quantity"},
    {"id": "R37", "kind": "tricky", "qty": 200, "deadline_days": 21,
     "text": "200 cft of kiln-dried Dalbergia sissoo, 1.5 inch, moisture up to 12%, to Jodhpur within 3 weeks.",
     "expected": "synonym: Dalbergia sissoo is sheesham",
     "match": lambda p: p["product"] == "timber" and p["species"] == "sheesham" and p["seasoning"] == "kiln-dried"
     and p["moisture_pct"][1] <= 12 and 1.5 in p["thickness_in"]},
    {"id": "R38", "kind": "tricky", "qty": 300, "deadline_days": 28,
     "text": "300 m of fire-retardant sofa fabric for a hotel lobby in Udaipur, within 4 weeks.",
     "expected": "synonym: fire-retardant = flame-retardant; stain-resistant is a near miss",
     "match": lambda p: p["product"] == "fabric" and p["flame_retardant"]},
    {"id": "R39", "kind": "tricky", "qty": 100, "deadline_days": 20,
     "text": "100 sheets of 30 mm waterproof plywood, delivered to Jodhpur within 20 days.",
     "expected": "no vendor stocks 30 mm; report the closest (25 mm)",
     "match": lambda p: p["product"] == "panel" and p["board"] == "plywood_wp" and 30 in p["thickness_mm"]},
    {"id": "R40", "kind": "tricky", "qty": 50, "deadline_days": 2,
     "text": "50 kg of memory foam, delivered to Jodhpur within 2 days.",
     "expected": "nobody delivers in 2 days; report the fastest",
     "match": lambda p: p["product"] == "foam" and p["type"] == "memory_foam"},
]
REQ = {r["id"]: r for r in SEED_REQUESTS}


def gold_for(req, vendors, truths):
    if "match" not in req:
        return []
    out = []
    for v, t in zip(vendors, truths, strict=True):
        if not any(req["match"](p) for p in t["products"]):
            continue
        if v["moq"] > req["qty"]:
            continue
        if req["deadline_days"] is not None and v["lead_time_days"] > req["deadline_days"]:
            continue
        if not serves_raj(v):
            continue
        out.append(v["vendor_id"])
    return out


# --------------------------------------------------------------------------------------
# Planted hard cases
# --------------------------------------------------------------------------------------
def perfect_product(rng, rid):
    return {
        "R01": lambda: p_timber(rng, species="sheesham", seasoning="kiln-dried", moisture=(10, 12),
                                thickness=sorted({1.5, *rng.sample([1, 2, 2.5], 2)})),
        "R02": lambda: p_panel(rng, kind="plywood_wp", thickness=[12, 16, 18]),
        "R03": lambda: p_hw(rng, kind="drawer_channel", soft_close=True, lengths=[350, 400, 450, 500]),
        "R04": lambda: p_foam(rng, kind="pu_foam", densities=[32, 40, 50]),
        "R05": lambda: p_fabric(rng, kind="fr_fabric"),
        "R06": lambda: p_panel(rng, kind="prelam_mdf", thickness=[16, 18], finishes=["walnut", "oak", "white"]),
        "R07": lambda: p_pack(rng, kind="box", ply=[5], custom=True),
    }.get(rid, lambda: NEW_PERFECT[rid](rng))()


def _with(p, **attrs):
    p["attrs"].update(attrs)
    return p


NEW_PERFECT = {
    "R13": lambda r: _with(p_hw(r, kind="hinge", soft_close=True), overlay=["full", "half"]),
    "R14": lambda r: _with(p_hw(r, kind="gas_lift"), use="bed", force_n=1000),
    "R15": lambda r: _with(p_lam(r, kind="laminate_1mm"), textures=["gloss", "matte", "suede"]),
    "R16": lambda r: _with(p_lam(r, kind="veneer"), species=["oak", "teak", "walnut"]),
    "R17": lambda r: p_foam(r, kind="hr_foam", densities=[40, 50]),
    "R18": lambda r: _with(p_fabric(r, kind="leatherette"), base="PU"),
    "R19": lambda r: _with(p_adh(r, kind="pu_lacquer"), sheen="matte"),
    "R20": lambda r: p_adh(r, kind="pva_glue"),
    "R21": lambda r: _with(p_metal(r, kind="ss_legs"), custom_drawings=True),
    "R22": lambda r: _with(p_edge(r, kind="pvc_edge"), thickness_mm=[0.8, 2.0], width_mm=[22, 45]),
    "R23": lambda r: _with(p_glass(r, kind="toughened"), thickness_mm=[8, 10, 12]),
    "R24": lambda r: _with(p_glass(r, kind="mirror"), thickness_mm=[4, 5]),
    "R25": lambda r: p_serv(r, kind="cnc_routing"),
    "R26": lambda r: p_timber(r, species="teak", seasoning="kiln-dried", moisture=(10, 12), thickness=[1.5, 2, 2.5]),
    "R27": lambda r: p_timber(r, species="mango", seasoning="kiln-dried", moisture=(8, 10), thickness=[1, 1.5, 2]),
    "R28": lambda r: p_panel(r, kind="hdhmr", thickness=[12, 16, 18]),
    "R29": lambda r: _with(p_pack(r, kind="stretch_wrap"), micron=23),
    "R30": lambda r: p_foam(r, kind="memory_foam"),
    "R31": lambda r: p_hw(r, kind="tandem", lengths=[400, 450, 500, 550]),
}

CAT_OF = {"R01": TIMBER, "R02": PANEL, "R03": HW, "R04": FOAM, "R05": FABRIC, "R06": PANEL, "R07": PACK,
          "R13": HW, "R14": HW, "R15": LAM, "R16": LAM, "R17": FOAM, "R18": FABRIC, "R19": ADH, "R20": ADH,
          "R21": METAL, "R22": EDGE, "R23": GLASS, "R24": GLASS, "R25": SERV, "R26": TIMBER, "R27": TIMBER,
          "R28": PANEL, "R29": PACK, "R30": FOAM, "R31": HW}
# genuine full matches planted per clear request (the original 7 get 5, the week-3 ones 3)
ANCHORS = {rid: (5 if rid in ("R01", "R02", "R03", "R04", "R05", "R06", "R07") else 3) for rid in CAT_OF}
# category of every request, for the eval's hand-labelled filters (vague ones included)
REQUEST_CATEGORY = CAT_OF | {"R08": TIMBER, "R09": HW, "R10": FABRIC, "R11": TIMBER, "R12": PANEL,
                             "R32": FOAM, "R33": GLASS, "R34": LAM, "R35": ADH, "R36": PACK,
                             "R37": TIMBER, "R38": FABRIC, "R39": PANEL, "R40": FOAM}

NEAR_MISS = [  # (request, count, product factory, note)
    ("R01", 3, lambda r: p_timber(r, species="sheesham", seasoning="air-dried"), "air-dried, not kiln-dried"),
    ("R01", 2, lambda r: p_timber(r, species="sheesham", seasoning="kiln-dried", moisture=(12, 14), thickness=[1, 1.5, 2]),
     "moisture 12–14%, above the 12% limit"),
    ("R01", 1, lambda r: p_timber(r, species="sheesham", seasoning="kiln-dried", moisture=(10, 12), thickness=[2, 2.5, 3]),
     "no 1.5 inch thickness"),
    ("R02", 3, lambda r: p_panel(r, kind="plywood_mr", thickness=[12, 16, 18]), "moisture-resistant, not waterproof"),
    ("R02", 1, lambda r: p_panel(r, kind="plywood_wp", thickness=[12, 16]), "waterproof, but no 18 mm"),
    ("R03", 3, lambda r: p_hw(r, kind="drawer_channel", soft_close=False, lengths=[400, 450, 500]), "not soft-close"),
    ("R03", 1, lambda r: p_hw(r, kind="drawer_channel", soft_close=True, lengths=[300, 350, 400]), "no 450 mm length"),
    ("R04", 3, lambda r: p_foam(r, kind="pu_foam", densities=[23, 28, 32]), "max density 32, needs 40"),
    ("R05", 2, lambda r: p_fabric(r, kind="stain_resistant"), "stain-resistant, not flame-retardant"),
    ("R06", 1, lambda r: p_panel(r, kind="prelam_mdf", thickness=[12, 16], finishes=["walnut", "wenge"]), "no 18 mm"),
    ("R06", 1, lambda r: p_panel(r, kind="prelam_particle", thickness=[16, 18], finishes=["walnut", "white"]),
     "particle board, not MDF"),
    ("R07", 1, lambda r: p_pack(r, kind="box", ply=[3], custom=True), "3-ply only, needs 5-ply"),
]

KEYWORD_TRAPS = [  # (category, target request, spec text, note)
    (PANEL, "R02", "Moisture-resistant plywood, 12, 16 and 18 mm, 8x4 ft sheets, interior use. Also stock waterproof "
     "tarpaulins and HDPE covers for site storage.", "'waterproof' refers to tarpaulins; the plywood is only moisture-resistant"),
    (LAM, "R01", "1 mm decorative laminates: sheesham-look, teak-look and walnut-look woodgrain designs, 8x4 ft.",
     "laminate with a sheesham look, not sheesham timber"),
    (HW, "R03", "Telescopic ladders, telescopic curtain rods and 450 mm shelf brackets.",
     "'telescopic' and '450 mm' but no drawer channels"),
    (PACK, "R04", "EPE packaging foam rolls, 20–40 mm thick, density 40 grade for heavy furniture.",
     "packaging foam, not PU seating foam"),
    (FABRIC, "R05", "Fire-safety blankets and flame-retardant welding curtains for factories.",
     "flame-retardant, but not upholstery fabric"),
    (ADH, "R06", "Walnut wood stain and MDF sealer, 1 L and 4 L cans.", "mentions walnut and MDF, sells finishes"),
    (SERV, "R01", "Kiln drying job work for sheesham and acacia — bring your own timber, we season it to 10–12%.",
     "seasoning service, does not sell timber"),
    (GLASS, "R01", "Mirrors with carved sheesham frames, 2x3 ft and 3x4 ft.", "mirror, not timber"),
]

VAGUE = [
    (TIMBER, "All types of wood available at best rates. Contact for details."),
    (PANEL, "Dealing in all kinds of plywood and boards. Best quality guaranteed."),
    (HW, "Complete furniture hardware solutions under one roof."),
    (FOAM, "All types of foam available."),
    (FABRIC, "Sofa fabric, curtain fabric, all varieties available at wholesale price."),
    (PACK, "Premium quality packaging material for all industries."),
    (PANEL, "Supplier of all furniture raw material."),
    (LAM, "Laminates, veneers and more. Premium quality material."),
]


def name_variant(name):
    for a, b in [("Shree", "Shri"), ("Shri", "Shree"), (" Pvt. Ltd.", ""), (" & Sons", " and Sons"),
                 (" & Co.", " and Company"), (" (India)", "")]:
        if a in name:
            return name.replace(a, b, 1)
    return name + " (Unit 2)"


def generate(n, seed, email_base):
    f = Factory(seed, email_base)
    rng = f.rng
    vendors, truths, hard = [], [], []

    def add(v, t, case=None, target=None, note=""):
        t["hard_case"], t["target_request"] = case, target
        vendors.append(v)
        truths.append(t)
        if case:
            hard.append({"vendor_id": v["vendor_id"], "name": v["name"], "case_type": case,
                         "target_request": target or "", "note": note})

    planted = (sum(c for _, c, _, _ in NEAR_MISS) + 15 + len(KEYWORD_TRAPS) + len(VAGUE) + 6
               + sum(ANCHORS.values()))
    n_base = n - planted

    # 1) base vendors
    cats = list(CATEGORY_WEIGHTS)
    for _ in range(n_base):
        cat = rng.choices(cats, [CATEGORY_WEIGHTS[c] for c in cats])[0]
        v, t = f.build(cat)
        add(v, t)

    # 2) synonyms: rewrite ~14 timber/panel vendors using alternative names (they still match)
    syn_pool = [i for i, t in enumerate(truths)
                if any(p["product"] == "timber" and p["species"] in ("sheesham", "teak", "mango") or
                       p["product"] == "panel" and p["board"] in ("hdhmr", "mdf", "prelam_mdf", "particle")
                       for p in t["products"])]
    for i in rng.sample(syn_pool, min(14, len(syn_pool))):
        prods = [dict(product=p["product"], attrs={k: v for k, v in p.items() if k != "product"}) for p in truths[i]["products"]]
        vendors[i]["specs"] = " ".join(describe(rng, p, synonym=True) for p in prods)
        truths[i]["hard_case"] = "synonym"
        hard.append({"vendor_id": vendors[i]["vendor_id"], "name": vendors[i]["name"], "case_type": "synonym",
                     "target_request": "", "note": "uses an alternative name for the material"})

    # 3) near-miss specs: constraints all pass, only the spec fails
    for rid, count, factory, note in NEAR_MISS:
        req = REQ[rid]
        for _ in range(count):
            dl = req["deadline_days"] or 20
            v, t = f.build(CAT_OF[rid], products=[factory(rng)], state_region="Rajasthan",
                           moq=nice_round(req["qty"] * rng.uniform(0.2, 0.6)),
                           lead=rng.randint(max(2, dl - 8), dl), regions=["Rajasthan", "Gujarat"])
            add(v, t, "near_miss", rid, note)

    # 4) right product, wrong constraint
    fails = ["moq", "lead", "region"]
    rids = ["R01", "R02", "R03", "R04", "R05", "R07", "R06"]
    for k in range(15):
        rid = rids[k % len(rids)]
        req = REQ[rid]
        fail = fails[k % 3] if req["deadline_days"] else ["moq", "region"][k % 2]
        dl = req["deadline_days"] or 20
        kw = dict(moq=nice_round(req["qty"] * rng.uniform(0.2, 0.6)),
                  lead=rng.randint(max(2, dl - 8), dl), regions=["Rajasthan"], state_region="Rajasthan")
        if rid == "R02":
            kw["lead"] = max(kw["lead"], 7)
        if fail == "moq":
            kw["moq"] = nice_round(req["qty"] * rng.uniform(1.5, 3))
            note = f"MOQ above the requested {req['qty']}"
        elif fail == "lead":
            kw["lead"] = dl + rng.randint(3, 15)
            note = f"lead time over the {dl}-day deadline"
        else:
            reg = rng.choice(["Maharashtra", "South India"])
            kw["state_region"] = reg
            state = CITIES[reg][0][1]
            kw["regions"] = [state] + [s for s in NEIGHBOURS[state] if s != "Rajasthan"][:2]
            note = "does not deliver to Rajasthan"
        v, t = f.build(CAT_OF[rid], products=[perfect_product(rng, rid)], **kw)
        if fail == "region":
            v["regions_served"] = kw["regions"] = [v["state"]] + [s for s in NEIGHBOURS[v["state"]] if s != "Rajasthan"][:2]
        add(v, t, f"wrong_constraint:{fail}", rid, note)

    # 4b) anchors: a few genuine full matches per clear request, spread across regions,
    #     so every clear request has enough gold vendors for precision@5 to be meaningful
    for rid in CAT_OF:
        req = REQ[rid]
        dl = req["deadline_days"] or 25
        for _ in range(ANCHORS[rid]):
            reg = rng.choices(["Rajasthan", "Gujarat", "Delhi NCR", "UP & Haryana"], [55, 20, 15, 10])[0]
            v, t = f.build(CAT_OF[rid], products=[perfect_product(rng, rid)], state_region=reg,
                           moq=nice_round(req["qty"] * rng.uniform(0.1, 0.7)),
                           lead=rng.randint(max(7 if rid == "R02" else 2, dl - 10), dl))
            if not serves_raj(v):
                v["regions_served"] = v["regions_served"] + ["Rajasthan"]
            add(v, t)
            t["target_request"] = rid

    # 5) keyword traps
    for cat, rid, text, note in KEYWORD_TRAPS:
        v, t = f.build(cat, products=[{"product": "other", "attrs": {"description": text}, "price": GEN[cat](rng)["price"],
                                        "price_unit": GEN[cat](rng)["price_unit"]}],
                       text_override=text, state_region="Rajasthan", regions=["Rajasthan"],
                       moq=MOQ_RANGE[cat][0], lead=LEAD_RANGE[cat][0])
        add(v, t, "keyword_trap", rid, note)

    # 6) vague vendors
    for cat, text in VAGUE:
        v, t = f.build(cat, products=[{"product": "unspecified", "attrs": {}, "price": GEN[cat](rng)["price"],
                                        "price_unit": GEN[cat](rng)["price_unit"]}], text_override=text)
        add(v, t, "vague", None, "no concrete specs")

    # 7) duplicates of base vendors
    for i in rng.sample(range(n_base), 6):
        src_v, src_t = vendors[i], truths[i]
        v = copy.deepcopy(src_v)
        t = copy.deepcopy(src_t)
        vid = f.new_id()
        v["vendor_id"] = t["vendor_id"] = vid
        v["name"] = name_variant(src_v["name"])
        v["contact_email"] = f"{f.local}+{vid.lower().replace('-', '')}@{f.domain}"
        add(v, t, "duplicate", None, f"duplicate of {src_v['vendor_id']} ({src_v['name']})")

    # shuffle so planted vendors are not clustered at the end, then renumber ids
    order = list(range(len(vendors)))
    rng.shuffle(order)
    vendors = [vendors[i] for i in order]
    truths = [truths[i] for i in order]
    remap = {}
    for k, (v, t) in enumerate(zip(vendors, truths, strict=True), start=1):
        new = f"V-{k:04d}"
        remap[v["vendor_id"]] = new
        v["vendor_id"] = t["vendor_id"] = new
        v["contact_email"] = f"{f.local}+{new.lower().replace('-', '')}@{f.domain}"
        v["phone"] = f"+91-00000-{k:05d}"
    for h in hard:
        h["vendor_id"] = remap[h["vendor_id"]]
        if h["case_type"] == "duplicate":
            old = h["note"].split()[2]
            h["note"] = h["note"].replace(old, remap[old])
    hard.sort(key=lambda h: h["vendor_id"])
    return vendors, truths, hard


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
CSV_FIELDS = ["vendor_id", "name", "city", "state", "regions_served", "categories", "specs", "moq", "moq_unit",
              "lead_time_days", "price_band", "indicative_price", "payment_terms", "gst_registered",
              "rating_notes", "contact_name", "contact_email", "phone"]


def write_all(out, eval_out, vendors, truths, hard):
    out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)
    with open(out / "vendors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for v in vendors:
            row = dict(v)
            row["regions_served"] = " | ".join(v["regions_served"])
            row["categories"] = " | ".join(v["categories"])
            w.writerow(row)
    (out / "vendors.json").write_text(json.dumps(vendors, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(eval_out / "vendor_truth.jsonl", "w", encoding="utf-8") as fh:
        for t in truths:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(eval_out / "hard_cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["vendor_id", "name", "case_type", "target_request", "note"])
        w.writeheader()
        w.writerows(hard)
    reqs = []
    for r in SEED_REQUESTS:
        category = REQUEST_CATEGORY[r["id"]]
        reqs.append({k: v for k, v in r.items() if k != "match"} |
                    {"category": category, "delivery_state": "Rajasthan",
                     "gold_vendor_ids": gold_for(r, vendors, truths) if "match" in r else []})
    (eval_out / "seed_requests.json").write_text(json.dumps(reqs, indent=2, ensure_ascii=False), encoding="utf-8")
    return reqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--email-base", default="rfq.test@example.com",
                    help="a test inbox YOU control; each vendor gets a +tag address on it")
    ap.add_argument("--out", default=".", help="where vendors.csv / vendors.json go")
    ap.add_argument("--eval-out", default=None, help="where the eval-only files go (default: same as --out)")
    a = ap.parse_args()
    vendors, truths, hard = generate(a.n, a.seed, a.email_base)
    reqs = write_all(Path(a.out), Path(a.eval_out or a.out), vendors, truths, hard)

    print(f"Wrote {len(vendors)} vendors ({len(hard)} planted hard-case labels) to {a.out}")
    for r in reqs:
        g = r["gold_vendor_ids"]
        print(f"  {r['id']} [{r['kind']:6}] gold matches: {len(g):3d}  {r['text'][:70]}")


if __name__ == "__main__":
    main()
