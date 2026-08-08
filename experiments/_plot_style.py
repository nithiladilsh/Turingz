import matplotlib

PALETTE = {
    "indigo": "#4f46e5",
    "indigo_light": "#c7d2fe",
    "emerald": "#059669",
    "rose": "#e11d48",
    "amber": "#d97706",
    "violet": "#7c3aed",
    "grid": "#eef0f3",
    "axis": "#64748b",
    "ink": "#1e2430",
}


def apply_app_style():
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["axis"],
        "xtick.color": PALETTE["axis"],
        "ytick.color": PALETTE["axis"],
        "text.color": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "grid.color": PALETTE["grid"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
    })
