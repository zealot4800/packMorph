# PackMorph deep-learning simulation

This repository contains only the simulation portion of PackMorph. It replaces the paper's Random Forest classifier with a padding-aware dilated temporal CNN and excludes the Mininet, TUN/TAP, Open vSwitch, and live sender/receiver prototype.

The experiment classifies five services in the supplied VPN and non-VPN captures: Streaming, VoIP, Chat, Command and Control (`C2`), and File Transfer.

## Method

1. `tshark` extracts bidirectional TCP/UDP packet sequences from PCAP or PCAPNG files.
2. Capture-grouped splits keep every capture entirely within training, validation, or evaluation.
3. The default `balanced_guided` policy trains two independently seeded selectors and a separate evaluator. Candidate utility is the worst result across the selector ensemble; the evaluator is not queried during selection and produces the reported accuracies.
4. Guided selection maximizes robust assigned-target confidence and target-versus-source odds. Among candidates within the configured utility tolerance, it prefers the lower weighted Chebyshev overhead cost. Morphing preserves directional source-byte capacity, enforces the packet-size and per-sequence overhead limits, and re-windows every generated packet before prediction.
5. Only four aggregate results are published for each environment.

## Installation

Python dependencies are defined in `pyproject.toml`; `requirements.txt` installs the project in editable mode. `tshark` must also be installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default `model.device: cpu` disables GPU use. Set it to `auto` to use an available TensorFlow GPU with CPU fallback, or to `gpu` to require a GPU. On Linux or WSL2, TensorFlow's CUDA dependencies can be installed with:

```bash
python -m pip install "tensorflow[and-cuda]>=2.15,<2.21"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Training uses bounded eager batches and conservative thread limits for CPU/GPU stability. Selector training, evaluator training, and final evaluation run in disposable processes so a native TensorFlow failure can be retried cleanly.

## Running

Extract the configured captures and run both environments:

```bash
packmorph-sim all --config configs/default.yaml
```

Use `--force-extract` to rebuild the sequence cache or `--verbose` for detailed extraction and error messages. Extraction and evaluation can also run separately:

```bash
packmorph-sim extract --config configs/default.yaml
packmorph-sim run --config configs/default.yaml
```

Without installing the package, use `python3 -m src` from the repository root. Relative paths are resolved from the YAML file's directory.

## Results

The current default experiment produces:

| Environment | Baseline accuracy | Morphed accuracy | Bandwidth overhead | FCT overhead |
|---|---:|---:|---:|---:|
| Non-VPN | 93.20% | 4.79% | 9.59% | 2.98% |
| VPN | 92.21% | 21.06% | 12.86% | 2.31% |

The experiment output contains exactly one file:

```text
artifacts/experiment/
└── summary.json
```

Its strict format is:

```json
{
  "Non VPN": {
    "baseline_accuracy": 0.9319938176197836,
    "morphed_accuracy": 0.04791344667697063,
    "bandwidth_overhead": 0.09587221045127484,
    "fct_overhead": 0.029829984544049455
  },
  "VPN": {
    "baseline_accuracy": 0.9221238938053097,
    "morphed_accuracy": 0.21061946902654868,
    "bandwidth_overhead": 0.1285907534944062,
    "fct_overhead": 0.023062365466627082
  }
}
```

All values are ratios: `0.03` means 3%. Baseline and morphed accuracy are measured on the same evaluation sequences. Bandwidth overhead is the mean transmitted-byte increase per source sequence. FCT overhead is the mean duration increase over source sequences with positive duration. The configured bandwidth and FCT limits are enforced per sequence; the table shows their means.

Lower morphed accuracy means the evaluator retained the original service label less often; it does not by itself prove successful classification as the assigned target service. These are exploratory same-split development results because the earlier results informed the new scoring and padding search. Confirm the frozen method on fresh capture-disjoint outer test groups before treating the numbers as final paper results.

Checkpoints and guided-selection arrays are temporary implementation files. They are validated during execution and deleted before the completed `summary.json` is published. A failed run preserves the previous complete output.

## Configuration

[configs/default.yaml](configs/default.yaml) controls extraction, capture-grouped splits, model capacity, CPU/GPU selection, target policy, and the bandwidth/FCT limits.

The default uses the smaller local `dataset_10` tree and retains at most 200 windows per capture. For a full-corpus run, set `data.root` to `../data` and `max_windows_per_capture` to `null`. The sequence cache is stored separately at `artifacts/sequences.npz`; it is an extraction input to the experiment, not a reported result.

Available target policies are `balanced_guided`, `balanced`, `random`, and `nearest`. For each training-only target profile, the default guided policy evaluates `padding_fractions` and `fct_fractions` as fractions of the configured per-sequence allowances. A value of `0` uses no optional allowance and `1` permits the full corresponding cap. Attack utility combines absolute assigned-target confidence with target-versus-source odds, preventing a negligible target probability from appearing successful merely because the source probability is even smaller. The worst utility across `model.selector_count` independently seeded selectors is used to reduce selector-specific overfitting. Candidates within `attack_utility_tolerance` of the best robust utility minimize their maximum normalized resource cost after bandwidth and FCT are scaled by `bandwidth_cost_weight` and `fct_cost_weight`; a resource-sum tie-break prevents dominated choices on Chebyshev plateaus. At least one resource-cost weight must be positive. An exact quota-constrained assignment then distributes each source service as evenly as possible across non-source services. This balances selected source-sequence targets; it does not guarantee equal evaluator predictions.

## Scope

The paper uses CICFlowMeter aggregate features, whereas this implementation intentionally uses raw packet sequences for the temporal CNN. It is therefore a deep-learning simulation extension, not an exact numerical reproduction of the paper.

The simulator accounts for opaque bytes and timing but does not transport real payloads. Network loss, congestion, retransmissions, synchronization metadata, and application-visible correctness belong to the excluded prototype phase. Guided assignment is an offline simulation that assumes the source service is known and plans targets over the evaluation partition.

## License

Licensed under the [Apache License 2.0](LICENSE).
