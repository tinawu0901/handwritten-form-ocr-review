"""Process one photographed form with OpenCV and PaddleOCR PPStructure."""
import argparse
import json
from pathlib import Path

from preprocessing import crop_document, preprocess, read_image, write_image

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    source = args.input.resolve()
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    output = args.output.resolve() if args.output else ROOT / "runs" / source.stem
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose another --output: {output}")
    image = read_image(source)
    cropped, detected = (image.copy(), False) if args.no_crop else crop_document(image)
    enhanced = preprocess(cropped, args.preprocess)
    output.mkdir(parents=True, exist_ok=True)
    write_image(output / "cropped.png", cropped)
    write_image(output / "ocr-input.png", enhanced)
    report = {"input": str(source), "document_detected": detected,
              "crop_enabled": not args.no_crop, "preprocessing": args.preprocess}
    if not args.preprocess_only:
        from paddleocr import PPStructure, save_structure_res
        models = ROOT / ".models"
        engine = PPStructure(
            lang="ch", use_gpu=False, show_log=False, layout=False,
            use_angle_cls=True, enable_mkldnn=False, image_orientation=False,
            det_model_dir=str(models / "det"), rec_model_dir=str(models / "rec"),
            cls_model_dir=str(models / "cls"), table_model_dir=str(models / "table-2.9"),
        )
        # Pass the actual processed pixels, not the original file path.
        result = engine(enhanced)
        save_structure_res(result, str(output), "table")
        regions = [{key: value for key, value in item.items() if key != "img"}
                   for item in result]
        report["regions"] = regions
        report["table_count"] = sum(item.get("type") == "table" for item in result)
    (output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   default=lambda value: value.tolist()), encoding="utf-8")
    print(f"Document detected: {detected}; preprocessing: {args.preprocess}")
    if args.preprocess_only:
        print("Preprocessing only: OCR was not run.")
    else:
        print(f"Table regions: {report['table_count']}; inspect generated Excel for recognition errors.")
    print(f"Output: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input form photograph")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--preprocess", choices=["none", "binary", "shadow"], default="none")
    parser.add_argument("--no-crop", action="store_true", help="Skip automatic document cropping")
    parser.add_argument("--preprocess-only", action="store_true", help="Save images without loading OCR models")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
