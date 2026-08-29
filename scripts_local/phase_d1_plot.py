"""Render dependency-free Phase D1 line charts with Pillow."""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from project_paths import load_config


CONFIG = load_config("configs/phase_d1_cpu_simulation.yaml")
OUTPUT_DIR = Path(CONFIG["output_dir"])
with (OUTPUT_DIR / "summary.csv").open(encoding="utf-8") as file:
    ROWS = list(csv.DictReader(file))

COLORS = {
    "largest_remainder": "#d95f02",
    "systematic_stochastic": "#1b9e77",
    "cumulative_deficit": "#7570b3",
}
LABELS = {
    "largest_remainder": "Largest remainder",
    "systematic_stochastic": "Stochastic systematic",
    "cumulative_deficit": "Cumulative deficit",
}
FONT = ImageFont.load_default()


def render(metric, title, ylabel, filename):
    width, height = 1500, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margins = (65, 85, 25, 55)
    panel_gap = 35
    panel_width = (width - margins[0] - margins[2] - 2 * panel_gap) / 3
    plot_height = height - margins[1] - margins[3]
    values = [float(row[metric]) for row in ROWS]
    ymax = max(values) * 1.08 if max(values) > 0 else 1.0
    ymin = 0.0
    draw.text((65, 14), title, fill="black", font=FONT)

    for panel_index, candidate_count in enumerate(CONFIG["candidate_counts"]):
        left = margins[0] + panel_index * (panel_width + panel_gap)
        top = margins[1]
        right = left + panel_width
        bottom = top + plot_height
        draw.line((left, top, left, bottom), fill="#444444", width=1)
        draw.line((left, bottom, right, bottom), fill="#444444", width=1)
        for tick in range(6):
            value = ymin + (ymax - ymin) * tick / 5
            y = bottom - plot_height * tick / 5
            draw.line((left, y, right, y), fill="#e5e5e5", width=1)
            draw.text((left - 54, y - 5), f"{value:.2f}", fill="#444444", font=FONT)
        budgets = CONFIG["budgets"]
        for index, budget in enumerate(budgets):
            x = left + panel_width * index / (len(budgets) - 1)
            draw.line((x, bottom, x, bottom + 4), fill="#444444", width=1)
            draw.text((x - 7, bottom + 8), str(budget), fill="#444444", font=FONT)
        draw.text((left + panel_width / 2 - 28, bottom + 28), "Budget N", fill="black", font=FONT)
        draw.text((left + panel_width / 2 - 55, top - 20), f"Candidates K={candidate_count}", fill="black", font=FONT)

        for method, color in COLORS.items():
            selected = {
                int(row["budget"]): float(row[metric])
                for row in ROWS
                if int(row["candidate_count"]) == candidate_count and row["method"] == method
            }
            points = []
            for index, budget in enumerate(budgets):
                x = left + panel_width * index / (len(budgets) - 1)
                y = bottom - (selected[budget] - ymin) / (ymax - ymin) * plot_height
                points.append((x, y))
            draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    draw.text((8, height / 2 - len(ylabel) * 3), ylabel, fill="black", font=FONT)
    legend_x = width - 525
    for index, (method, color) in enumerate(COLORS.items()):
        x = legend_x + index * 175
        draw.line((x, 38, x + 22, 38), fill=color, width=3)
        draw.text((x + 27, 33), LABELS[method], fill="black", font=FONT)
    image.save(OUTPUT_DIR / filename)


render(
    "signal_erasure_rate_mean",
    "How often integer allocation erases a changed DORA signal",
    "Erasure rate",
    "signal_erasure_by_budget.png",
)
render(
    "final_cumulative_allocation_tv_error_mean",
    "Long-run deviation from the continuous allocation target",
    "Cumulative TV error",
    "cumulative_error_by_budget.png",
)
render(
    "mean_instantaneous_allocation_tv_error_mean",
    "Per-round deviation from the continuous allocation target",
    "Per-round TV error",
    "instantaneous_error_by_budget.png",
)
print(f"plots written to {OUTPUT_DIR}")
