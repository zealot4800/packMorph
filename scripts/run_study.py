"""Write all five-method comparison results to one CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import yaml

# Configure native-library noise before importing the simulation package.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from packmorph.comparison import run_comparison, validate_comparison_config
from packmorph.config import validate_base_config


FIELDS = (
    "run_id", "environment", "seed", "method", "point", "adversary",
    "samples", "accuracy", "macro_f1", "added_bytes_over_original_bytes",
    "packet_count_overhead", "mean_sequence_duration_overhead",
    "bandwidth_budget_exceeded_records", "duration_budget_exceeded_records",
    "incomplete_records", "fallback_records", "deadline_missed_flows",
)

LEGACY_FIELDS = (
    "run_id", "capture_fingerprint", "environment", "seed", "method", "point", "adversary",
    "samples", "accuracy", "macro_f1", "added_bytes_over_original_bytes",
    "packet_count_overhead", "dummy_packets", "dummy_bytes", "padding_bytes",
    "confusion_matrix", "metrics_json", "parameters_json",
    "validation_selection_json", "split_json", "model_settings_json", "data_settings_json",
    "excluded_captures_json",
)


QOS_FIELDS = (
    "run_id", "seed", "method", "point", "environment", "capture",
    "flow_group", "class", "transform_status", "policy_switches",
    "deadline_bypass_packets", "deadline_seconds", "baseline_completion_seconds",
    "completion_seconds", "deadline_met", "baseline_deadline_met",
    "lateness_seconds", "trace_goodput_bytes_per_second", "bandwidth_overhead",
    "duration_overhead", "duration_budget_met", "bandwidth_budget_met",
)


def _row(result: dict, run_id: str) -> dict:
    metrics = result["metrics"]
    return {
        "run_id": run_id,
        "environment": result["environment"], "seed": result["seed"],
        "method": result["method"], "point": result["point"], "adversary": result["adversary"],
        "samples": metrics["samples"], "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "added_bytes_over_original_bytes": metrics["added_bytes_over_original_bytes"],
        "packet_count_overhead": metrics["packet_count_overhead"],
        "mean_sequence_duration_overhead": metrics["sequence_duration_overhead"]["mean"],
        "bandwidth_budget_exceeded_records": metrics["bandwidth_budget_exceeded_records"],
        "duration_budget_exceeded_records": metrics["duration_budget_exceeded_records"],
        "incomplete_records": metrics["incomplete_records"],
        "fallback_records": metrics["fallback_records"],
        "deadline_missed_flows": metrics.get("qos", {}).get("deadline_missed_flows", ""),
    }


def _existing_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as previous:
        reader = csv.DictReader(previous)
        columns = tuple(reader.fieldnames or ())
        if columns not in (FIELDS, LEGACY_FIELDS):
            raise ValueError("Existing comparison CSV has an unsupported column schema")
        for row in reader:
            if columns == FIELDS:
                yield row
            else:
                metrics = json.loads(row["metrics_json"])
                yield _row({"environment": row["environment"], "seed": row["seed"],
                            "method": row["method"], "point": row["point"],
                            "adversary": row["adversary"], "metrics": metrics}, row["run_id"])


def load_study_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    required = {"output_csv", "data", "split", "model", "seeds", "budgets", "methods"}
    if not isinstance(settings, dict) or not required <= set(settings) or set(settings) - required - {"qos"}:
        raise ValueError(f"Study config requires {', '.join(sorted(required))}")
    output = Path(settings.pop("output_csv")).expanduser()
    if not output.is_absolute():
        output = path.parent / output
    if output.suffix.lower() != ".csv":
        raise ValueError("output_csv must name a CSV file")
    if "qos" in settings and isinstance(settings["qos"], dict):
        contracts = settings["qos"].get("flow_contracts_csv")
        if isinstance(contracts, str) and contracts:
            location = Path(contracts).expanduser()
            settings["qos"]["flow_contracts_csv"] = str((path.parent / location if not location.is_absolute()
                                                        else location).resolve())
    base = validate_base_config({key: settings.pop(key) for key in ("data", "split", "model")}, path)
    config = validate_comparison_config(settings, base)
    config["output_csv"] = str(output.resolve())
    return config


def write_results(config: dict) -> tuple[Path, int]:
    """Append comparison rows; write contracted-flow outcomes to a run-specific CSV."""
    destination = Path(config["output_csv"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    qos_destination = destination.with_name(f"{destination.stem}.qos.{run_id}.csv")
    temporary_path = qos_temporary_path = None
    count = qos_count = 0
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         prefix=".comparison-", suffix=".csv",
                                         dir=destination.parent, delete=False) as temporary, \
             tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         prefix=".qos-", suffix=".csv",
                                         dir=destination.parent, delete=False) as qos_temporary:
            temporary_path = Path(temporary.name)
            qos_temporary_path = Path(qos_temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=FIELDS)
            qos_writer = csv.DictWriter(qos_temporary, fieldnames=QOS_FIELDS)
            writer.writeheader()
            qos_writer.writeheader()
            if destination.exists():
                writer.writerows(_existing_rows(destination))
            for result in run_comparison(config):
                writer.writerow(_row(result, run_id))
                count += 1
                if result["adversary"] != "defense_aware":
                    for flow in result.get("qos_flows", ()):
                        qos_writer.writerow({
                            "run_id": run_id, "seed": result["seed"],
                            "method": result["method"], "point": result["point"], **flow,
                        })
                        qos_count += 1
        if qos_count:
            os.replace(qos_temporary_path, qos_destination)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if qos_temporary_path is not None:
            qos_temporary_path.unlink(missing_ok=True)
    return destination, count


def compact_existing(path: str | Path) -> tuple[Path, int]:
    """Replace a finished comparison CSV with its compact scalar summary."""
    destination = Path(path)
    if not destination.is_file():
        raise ValueError(f"Comparison CSV does not exist: {destination}")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         prefix=".comparison-", suffix=".csv",
                                         dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=FIELDS)
            writer.writeheader()
            count = 0
            for row in _existing_rows(destination):
                writer.writerow(row)
                count += 1
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination, count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the five-method packet-trace comparison")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--export-qos-template", metavar="CSV",
                        help="extract flow IDs into an edit-ready QoS contract CSV")
    parser.add_argument("--compact-existing", action="store_true",
                        help="compact a finished output CSV without rerunning the study")
    args = parser.parse_args()
    try:
        config = load_study_config(args.config)
        if args.compact_existing:
            path, rows = compact_existing(config["output_csv"])
            print(f"Compacted {rows} rows in {path}")
            return 0
        if args.export_qos_template:
            from packmorph.extract import extract_dataset
            from packmorph.qos import export_contract_template
            path = export_contract_template(extract_dataset(config["base"]),
                                            args.export_qos_template)
            print(f"Saved QoS contract template to {path}")
            return 0
        path, rows = write_results(config)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted\n")
    except Exception as error:
        parser.exit(1, f"Study failed: {error}\n")
    print(f"Saved {rows} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
