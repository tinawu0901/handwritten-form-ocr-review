"""Generate an invented printed form and intended values; no OCR or private inputs."""
import ast
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def schema():
    """Read only literal schema constants without importing the OCR application."""
    tree = ast.parse((ROOT / "python" / "cell_pipeline.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("COLUMNS", "HEADERS"):
                    values[target.id] = ast.literal_eval(node.value)
    return values["COLUMNS"], values["HEADERS"]


def main():
    columns, headers = schema()
    rows = [
        [str(i + 1), f"2/{i + 1}", f"AB{100 + i}", str(18 + i),
         "12", "15", "14", "8", "64", "", "-3", "", "", ""]
        for i in range(6)
    ]
    image = np.full((920, 2280, 3), 255, dtype=np.uint8)
    cv2.putText(image, "SYNTHETIC DEMO - PRINTED TEXT", (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, "Invented values only. Not a handwriting accuracy test.", (50, 115),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (80, 80, 80), 1, cv2.LINE_AA)
    left, top, width, row_height = 40, 170, 2200, 110
    xs = [left + round(c * width) for c in columns]
    for x in xs:
        cv2.line(image, (x, top), (x, top + 6 * row_height), (25, 25, 25), 3)
    for i in range(7):
        y = top + i * row_height
        cv2.line(image, (left, y), (left + width, y), (25, 25, 25), 3)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if text:
                cv2.putText(image, text, (xs[c] + 14, top + r * row_height + 68),
                            cv2.FONT_HERSHEY_SIMPLEX, .85, (110, 65, 25), 2, cv2.LINE_AA)
    examples = ROOT / "examples"
    examples.mkdir(exist_ok=True)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the synthetic form.")
    png.tofile(str(examples / "synthetic-form.png"))
    intended = {
        "description": "Invented printed demonstration data; intended values, not OCR output.",
        "headers": headers,
        "rows": rows,
        "data_area_corners": [
            [left - 4, top - 4], [left + width + 4, top - 4],
            [left + width + 4, top + 6 * row_height + 4],
            [left - 4, top + 6 * row_height + 4],
        ],
        "first_data_row": 1,
    }
    (examples / "expected.json").write_text(
        json.dumps(intended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Generated examples/synthetic-form.png and examples/expected.json")


if __name__ == "__main__":
    main()
