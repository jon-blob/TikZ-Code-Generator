import csv
from pathlib import Path

from config import PATHS


def normalize_path(value: str, base: Path, legacy_root: str, target_root: Path) -> str:
    value = value.removeprefix("file://")
    path = Path(value).expanduser()

    if path.is_absolute() and path.exists():
        return str(path)

    legacy = Path(legacy_root)
    if path.is_absolute() and path.is_relative_to(legacy):
        return str(target_root / path.relative_to(legacy))

    if not path.is_absolute():
        return str((base / path).resolve())

    return str(path)


def generate_tests(config=None):
    if not PATHS.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {PATHS.manifest}")

    base = PATHS.manifest.parent
    tests = []
    with PATHS.manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_image"):
                row["reference_image"] = normalize_path(row["reference_image"], base, "/images", PATHS.images)
            if row.get("reference_code"):
                row["reference_code"] = normalize_path(row["reference_code"], base, "/references", PATHS.references)
            if row.get("input_image"):
                row["input_image"] = normalize_path(row["input_image"], base, "/input_images", PATHS.references)
            tests.append({"vars": row})
    return tests
