"""Generate parquet dataset subsets from a benchmark run's per-sample metrics.

After running a full evaluation, this script partitions the test parquet into
quartile sub-datasets based on document_extraction_score (or any other metric).

Usage:
    python scripts/generate_dataset_subsets.py \\
        --metrics-file outputs/.../qwen3-vl-plus_..._per_sample_metrics.json \\
        --source-dataset ihsa-solicitud-extraction \\
        --output-prefix ihsa-solicitud-extraction

Outputs (under .data/):
    ihsa-solicitud-extraction-q4/   top quartile    (≥ Q3 threshold)
    ihsa-solicitud-extraction-q3/   upper-mid quartile
    ihsa-solicitud-extraction-q2/   lower-mid quartile
    ihsa-solicitud-extraction-q1/   bottom quartile (≤ Q1 threshold)

Each sub-dataset is a drop-in replacement: same schema.json, metrics_config.json,
and dataset_info.json (with updated row counts), and a new data/test.parquet.
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


BUCKET_ORDER = ["q4_top", "q3", "q2", "q1_bottom"]
BUCKET_SUFFIX = {
    "q4_top": "q4",
    "q3": "q3",
    "q2": "q2",
    "q1_bottom": "q1",
}
BUCKET_LABEL = {
    "q4_top": "Top quartile (highest scoring)",
    "q3": "Upper-mid quartile",
    "q2": "Lower-mid quartile",
    "q1_bottom": "Bottom quartile (lowest scoring)",
}


def load_entries(metrics_file: Path) -> list[dict]:
    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"No 'entries' found in {metrics_file}")
    return entries


def group_by_bucket(entries: list[dict], metric: str) -> dict[str, list]:
    """Group entries by performance_bucket, falling back to manual quartile split."""
    buckets: dict[str, list] = {b: [] for b in BUCKET_ORDER}

    if all("performance_bucket" in e for e in entries):
        for e in entries:
            bucket = e.get("performance_bucket")
            if bucket in buckets:
                buckets[bucket].append(e)
        if any(buckets.values()):
            return buckets

    # Fallback: manual quartile split on metric
    valid = [(e.get(metric), e) for e in entries if e.get(metric) is not None]
    valid.sort(key=lambda x: x[0])
    n = len(valid)
    q1_end = n // 4
    q2_end = n // 2
    q3_end = 3 * n // 4
    buckets["q1_bottom"] = [e for _, e in valid[:q1_end]]
    buckets["q2"] = [e for _, e in valid[q1_end:q2_end]]
    buckets["q3"] = [e for _, e in valid[q2_end:q3_end]]
    buckets["q4_top"] = [e for _, e in valid[q3_end:]]
    return buckets


def build_record_id_index(entries: list[dict]) -> dict[str, int]:
    """Build {record_id: parquet_row_index} mapping.

    Falls back to extracting the numeric suffix from sample_id
    (e.g. 'structured_42' → 42) when record_id is absent.
    """
    index: dict[str, int] = {}
    for e in entries:
        rid = e.get("record_id")
        sid = e.get("sample_id", "")
        if rid:
            index[rid] = None  # row resolved via record_id column in parquet
        elif sid and "_" in sid:
            try:
                row_idx = int(sid.rsplit("_", 1)[-1])
                index[f"__idx__{row_idx}"] = row_idx
            except ValueError:
                pass
    return index


def filter_parquet(
    table: pa.Table,
    entries: list[dict],
) -> pa.Table:
    """Return parquet rows that match the given entries."""
    # Build lookup: record_id → parquet row index
    if "record_id" in table.column_names:
        parquet_rids = [table.column("record_id")[i].as_py() for i in range(table.num_rows)]
        rid_to_idx = {rid: i for i, rid in enumerate(parquet_rids)}
    else:
        rid_to_idx = {}

    row_indices = []
    for e in entries:
        rid = e.get("record_id")
        sid = e.get("sample_id", "")

        if rid and rid in rid_to_idx:
            row_indices.append(rid_to_idx[rid])
        elif sid and "_" in sid:
            try:
                row_idx = int(sid.rsplit("_", 1)[-1])
                if 0 <= row_idx < table.num_rows:
                    row_indices.append(row_idx)
            except ValueError:
                pass

    if not row_indices:
        return table.slice(0, 0)  # empty table with same schema

    row_indices = sorted(set(row_indices))
    return table.take(row_indices)


def copy_dataset_metadata(src: Path, dst: Path, num_rows: int) -> None:
    """Copy schema.json, metrics_config.json, benchy.md; update dataset_info.json."""
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ("schema.json", "metrics_config.json", "benchy.md", "README.md", "manifest.json"):
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)

    info_path = src / "dataset_info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        info.setdefault("splits", {})
        info["splits"]["test"] = {"num_rows": num_rows}
        info["description"] = info.get("description", "") + f" [subset: {num_rows} rows]"
        with open(dst / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-file", required=True, help="Path to *_per_sample_metrics.json")
    parser.add_argument("--source-dataset", required=True, help="Dataset name under .data/ (e.g. ihsa-solicitud-extraction)")
    parser.add_argument("--output-prefix", default=None, help="Output dataset name prefix (default: source-dataset)")
    parser.add_argument("--metric", default="document_extraction_score", help="Metric to rank by (default: document_extraction_score)")
    parser.add_argument("--buckets", nargs="+", choices=list(BUCKET_ORDER), default=None, help="Buckets to generate (default: all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / ".data"
    source_dir = data_root / args.source_dataset
    parquet_path = source_dir / "data" / "test.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Source parquet not found: {parquet_path}")

    metrics_file = Path(args.metrics_file)
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    prefix = args.output_prefix or args.source_dataset
    target_buckets = set(args.buckets) if args.buckets else set(BUCKET_ORDER)

    print(f"Loading metrics from {metrics_file}")
    entries = load_entries(metrics_file)
    print(f"  {len(entries)} entries loaded")

    print(f"Loading parquet from {parquet_path}")
    table = pq.read_table(str(parquet_path))
    print(f"  {table.num_rows} rows, {len(table.column_names)} columns")

    buckets = group_by_bucket(entries, args.metric)

    # Print summary
    print(f"\nQuartile distribution (by {args.metric}):")
    for bucket in BUCKET_ORDER:
        entries_in_bucket = buckets[bucket]
        scores = [e.get(args.metric) for e in entries_in_bucket if e.get(args.metric) is not None]
        if scores:
            print(f"  {bucket:12s}: {len(entries_in_bucket):4d} samples  "
                  f"score {min(scores):.3f}–{max(scores):.3f}  "
                  f"mean {sum(scores)/len(scores):.3f}")

    print()
    for bucket in BUCKET_ORDER:
        if bucket not in target_buckets:
            continue
        bucket_entries = buckets[bucket]
        if not bucket_entries:
            print(f"  {bucket}: no entries, skipping")
            continue

        subset_table = filter_parquet(table, bucket_entries)
        suffix = BUCKET_SUFFIX[bucket]
        out_dir = data_root / f"{prefix}-{suffix}"
        out_parquet = out_dir / "data" / "test.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(subset_table, str(out_parquet))
        copy_dataset_metadata(source_dir, out_dir, subset_table.num_rows)

        scores = [e.get(args.metric) for e in bucket_entries if e.get(args.metric) is not None]
        mean_score = sum(scores) / len(scores) if scores else 0
        print(f"  Written {subset_table.num_rows} rows -> {out_dir.name}/  "
              f"(mean {args.metric}: {mean_score:.3f})")

    print("\nDone. Run smoke tests:")
    for bucket in BUCKET_ORDER:
        if bucket not in target_buckets:
            continue
        suffix = BUCKET_SUFFIX[bucket]
        print(f"  benchy eval --dataset-name {prefix}-{suffix} --task-type structured "
              f"--provider openai --model-name gpt-4o --limit 3")


if __name__ == "__main__":
    main()
