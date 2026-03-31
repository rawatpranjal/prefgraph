"""Generate hero diagram for FINN.no case study.

Shows a concrete example of a platform slate (menu) with translated
Norwegian category labels and the user's click (choice), illustrating
how menu-choice data maps to revealed preference analysis.

Follows PrefGraph visual style: #2563eb blue, #fafafa background, 150 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# PrefGraph style
BG = "#fafafa"
BLUE = "#2563eb"
LIGHT_BLUE = "#3b82f6"
RED = "#e74c3c"
DARK = "#333333"
SECONDARY = "#666666"
WHITE = "white"
DPI = 150

# Translation map for display
TRANSLATIONS = {
    "MOTOR": "Vehicles",
    "BAP": "Marketplace",
    "REAL_ESTATE": "Property",
    "JOB": "Jobs",
    "BOAT": "Boats",
    "TRAVEL": "Travel",
}

BAP_SUBS = {
    "antiques": "Antiques",
    "electronicsappliances": "Electronics",
    "entertainmenthobbyleisure": "Hobbies",
    "furnitureinterior": "Furniture",
    "housegardenrenovation": "Home & Garden",
    "clothescosmeticsaccessories": "Clothing",
    "sportsoutdoors": "Sports",
    "parentschildren": "Kids",
    "animalsequipment": "Pets",
    "business": "Business",
    "carmcboat": "Car Parts",
}

COUNTIES = {
    "Oslo": "Oslo",
    "Akershus": "Akershus",
    "Rogaland": "Rogaland",
    "Hordaland": "Bergen area",
    "Trøndelag": "Trondheim area",
    "Møre og Romsdal": "NW Coast",
    "Vestfold": "Vestfold",
    "Østfold": "SE Norway",
    "Nordland": "N. Norway",
    "Sogn og Fjordane": "Fjords",
    "Buskerud": "Buskerud",
    "Hedmark": "Hedmark",
    "Troms": "Tromsø area",
    "Finnmark": "Far North",
    "Telemark": "Telemark",
    "Vest-Agder": "S. Coast",
    "Aust-Agder": "S. Coast",
    "Oppland": "Oppland",
}


def translate_group(raw_label):
    """Translate a raw group label like 'BAP,furnitureinterior,Oslo' to English."""
    parts = raw_label.split(",")
    category = TRANSLATIONS.get(parts[0], parts[0])

    if parts[0] == "BAP" and len(parts) > 1 and parts[1]:
        sub = BAP_SUBS.get(parts[1], parts[1].title())
        category = sub

    county = ""
    if len(parts) > 2 and parts[2]:
        county = COUNTIES.get(parts[2], parts[2])
    elif len(parts) > 1 and parts[0] != "BAP" and parts[1]:
        county = COUNTIES.get(parts[1], parts[1])

    if county:
        return f"{category}, {county}"
    return category


def generate_hero():
    """Create a diagram showing one slate and the user's choice."""

    # Example slate: 5 items shown, user clicks one
    slate_items = [
        "Furniture\nOslo",
        "Vehicles\nRogaland",
        "Property\nBergen",
        "Electronics\nTrondheim",
        "Jobs\nOslo",
    ]
    chosen_idx = 2  # Property, Bergen

    fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.8, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Left side: slate
    ax.text(2.4, 3.6, "Slate shown to user", fontsize=9, color=SECONDARY,
            ha="center", va="top", style="italic")

    # Draw 5 item boxes in a row
    box_w, box_h = 0.85, 1.1
    gap = 0.12
    total_w = 5 * box_w + 4 * gap
    start_x = 2.4 - total_w / 2

    for i, label in enumerate(slate_items):
        x = start_x + i * (box_w + gap)
        y = 1.2
        is_chosen = (i == chosen_idx)

        color = BLUE if is_chosen else "#e8edf5"
        text_color = WHITE if is_chosen else DARK
        edge_color = BLUE if is_chosen else "#c0c8d8"
        lw = 2.5 if is_chosen else 1

        shadow = FancyBboxPatch(
            (x + 0.02, y - 0.02), box_w, box_h,
            boxstyle="round,pad=0.06", facecolor="#00000015", edgecolor="none",
        )
        ax.add_patch(shadow)

        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.06", facecolor=color, edgecolor=edge_color, linewidth=lw,
        )
        ax.add_patch(box)

        ax.text(x + box_w / 2, y + box_h / 2, label,
                fontsize=6.5, color=text_color, ha="center", va="center",
                fontweight="bold" if is_chosen else "normal", linespacing=1.3)

    # Click indicator arrow below the chosen box
    cx = start_x + chosen_idx * (box_w + gap) + box_w / 2
    ax.annotate("click", xy=(cx, 1.15), xytext=(cx, 0.55),
                fontsize=7, color=BLUE, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    # Arrow to right side
    ax.annotate("", xy=(5.6, 1.75), xytext=(5.1, 1.75),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))

    # Right side: revealed preference
    rx = 7.8
    ax.text(rx, 3.6, "Revealed preference", fontsize=9, color=SECONDARY,
            ha="center", va="top", style="italic")

    prefs = [
        ("Property, Bergen", True),
        ("\u227b  Furniture, Oslo", False),
        ("\u227b  Vehicles, Rogaland", False),
        ("\u227b  Electronics, Trondheim", False),
        ("\u227b  Jobs, Oslo", False),
    ]

    for i, (text, is_top) in enumerate(prefs):
        y_pos = 2.9 - i * 0.42
        if is_top:
            ax.text(rx, y_pos, text, fontsize=8.5, color=BLUE,
                    ha="center", va="center", fontweight="bold")
        else:
            ax.text(rx, y_pos, text, fontsize=7.5, color=SECONDARY,
                    ha="center", va="center")

    # Caption
    ax.text(5.0, -0.55,
            "The user saw 5 category-region groups and clicked Property, Bergen. "
            "This reveals a preference over the other 4 options.",
            fontsize=6.5, color=SECONDARY, ha="center", va="top", style="italic")

    plt.tight_layout()
    out = "case_studies/finn_slates/output/figures/fig0_hero.png"
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Also save to docs
    import shutil
    shutil.copy(out, "docs/_static/fig0_hero.png")
    print(f"Copied to docs/_static/fig0_hero.png")


if __name__ == "__main__":
    generate_hero()
