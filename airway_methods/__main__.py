"""Command-line interface for private local input and aggregate-only evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import SOURCE_COLUMNS, add_relative_features
from .models import CLINICAL, SCHEMES, evaluate


def extract(manifest: Path, output: Path):
    if output.exists():
        raise ValueError("Output already exists; refusing to overwrite")
    frame = pd.read_csv(manifest)
    required = {"patient_id", "a_path", "e_path", "i_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing manifest columns: " + ", ".join(missing))
    allowed = required | {"target", "age", *CLINICAL}
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError("Unexpected manifest columns (remove identifiers/extra fields): " + ", ".join(unexpected))
    if frame.patient_id.isna().any() or not frame.patient_id.is_unique:
        raise ValueError("One nonmissing patient_id per person is required")
    from .audio import extract_vowel
    rows = []
    for _, record in frame.iterrows():
        row = record.drop(labels=["a_path", "e_path", "i_path"]).to_dict()
        for label in "aei":
            path = Path(str(record[f"{label}_path"]))
            if not path.is_absolute():
                path = manifest.parent / path
            row.update({f"{label}_{key}": val for key, val in extract_vowel(path).items()})
        rows.append(row)
    result = add_relative_features(pd.DataFrame(rows))
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} local rows; this file contains patient-level data. Do not publish it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Print header names only, with no example patients")
    schema.add_argument("--scheme", choices=SCHEMES, required=True)
    extraction = commands.add_parser("extract", help="Extract features from privately held, presegmented WAVs")
    extraction.add_argument("--manifest", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    assessment = commands.add_parser("evaluate", help="Print aggregate internal CV metrics only")
    assessment.add_argument("--input", type=Path, required=True)
    assessment.add_argument("--scheme", choices=SCHEMES, required=True)
    assessment.add_argument("--folds", type=int, default=5)
    assessment.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.command == "schema":
        print(",".join(("patient_id", "target", *SCHEMES[args.scheme])))
    elif args.command == "extract":
        extract(args.manifest, args.output)
    else:
        result = evaluate(pd.read_csv(args.input), args.scheme, folds=args.folds, seed=args.seed)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
