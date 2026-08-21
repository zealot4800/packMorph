# PackMorph deep-learning simulation

This repository implements a **simulation-only deep-learning extension** of the supplied PackMorph paper. It replaces the paper's Random Forest adversary with a padding-aware **1D CNN + bidirectional GRU** and deliberately excludes Mininet, TUN/TAP, Open vSwitch, synchronization headers, and live sender/receiver code.

The supplied MIT Lincoln Laboratory VPN/non-VPN captures are evaluated as five service classes:

- Streaming
- VoIP
- Chat
- Command and Control (`C2`)
- File Transfer

## Simulation pipeline

1. `tshark` extracts bidirectional TCP/UDP flows from PCAP or PCAPNG files. Long flows become fixed-width packet windows; each packet carries length, inter-arrival time, direction, and a validity indicator.
2. Captures are split by class into disjoint training, validation, and evaluation partitions. Windows from one capture never cross partition boundaries.
3. A padding-aware CNN learns local packet and burst patterns, while a bidirectional GRU learns longer timing and direction dependencies. Class weighting limits bias from the dataset's uneven class sizes.
4. Offline profiles are learned only from the training partition. Every evaluation sequence selects a non-source decoy class and adopts a real profile's packet-size, IAT, direction, and volume behavior.
5. Fragmented transformed streams are re-windowed in full. Predictions across every generated window are averaged back to the original evaluation sequence, so classifier results and overhead statistics describe the same traffic.
6. The pipeline writes baseline/morphed confusion matrices, classification metrics, conditional evasion success, target-hit rate, byte/packet/FCT overhead, directional byte-capacity accounting, a model checkpoint, and source-level prediction arrays.

## Installation

Runtime dependencies are mirrored in `pyproject.toml` and `requirements.txt`. The runtime also needs the `tshark` executable.

```bash
python3 -m pip install -e .
```

The default `model.device: auto` uses a visible TensorFlow GPU when one is available and otherwise falls back to CPU. Set it to `cpu` to hide GPUs explicitly, or to `gpu` to require GPU acceleration and fail clearly when no usable GPU is found. GPU memory is allocated on demand rather than reserving all device memory at startup. The selected runtime device is printed and saved in each environment's `metrics.json`.

The normal installation above is sufficient for CPU execution. On Linux or WSL2, install TensorFlow's NVIDIA CUDA dependencies for GPU execution, then verify device discovery:

```bash
python3 -m pip install "tensorflow[and-cuda]>=2.15,<2.22"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

TensorFlow's oneDNN graph rewrites remain disabled for stability. Each VPN/non-VPN experiment also runs in a fresh process, preventing native TensorFlow state from leaking between environments on either backend.

## Running the experiment

Extract the configured PCAPs and run both VPN and non-VPN experiments:

```bash
packmorph-sim all --config configs/default.yaml
```

Force PCAP re-extraction:

```bash
packmorph-sim all --config configs/default.yaml --force-extract
```

Extraction and model execution can also run separately:

```bash
packmorph-sim extract --config configs/default.yaml
packmorph-sim run --config configs/default.yaml
```

Without installing the package, run `python3 -m src` from the repository root.

Relative dataset, cache, and output paths are resolved from the YAML file's directory. The default configuration therefore works independently of the shell's current directory.

## Outputs

```text
artifacts/experiment/
├── effective_config.yaml
├── summary.json
├── non_vpn/
│   ├── best_model.keras
│   ├── metrics.json
│   ├── predictions.npz
│   ├── morph_statistics.jsonl
│   ├── baseline_confusion.{csv,png}
│   ├── morphed_confusion.{csv,png}
│   └── training_history.png
└── vpn/
    └── ...
```

Important metrics are defined as follows:

- `conditional_evasion_success`: fraction of originally correct classifications made incorrect after morphing.
- `morphed_error_rate`: total error rate on transformed traffic, reported separately so baseline errors are not credited to PackMorph.
- `target_hit_rate`: fraction classified as the selected decoy class.
- `bandwidth_overhead`: transmitted byte increase relative to the source sequence.
- `fct_overhead`: transformed duration increase relative to the source sequence.
- `byte_accounting_accuracy`: fraction whose generated forward and reverse capacities account for every source byte in the same direction. This is not a payload reconstruction claim.

## Configuration and caching

[configs/default.yaml](configs/default.yaml) controls flow extraction, worker count, sequence width, partition fractions, model capacity, training, morphing policy, and overhead ceilings.

`max_windows_per_capture` retains a bounded prefix of packet windows from each capture. Set it to `null` to extract every available window.

The default uses the smaller local `dataset_10` capture tree for a bounded exploratory run. For a full-corpus study, set `data.root` to `../data` and `max_windows_per_capture` to `null`; expect substantially higher extraction time and storage use.

Available target policies are:

- `random`: independently selects a non-source class per source epoch, corresponding to periodically refreshed decoys.
- `nearest`: chooses the statistically closest non-source class to reduce transformation overhead.

Sequence caches contain a schema version and SHA-256 fingerprint over extraction settings and the PCAP manifest. A stale cache is rejected or automatically rebuilt by the `all` command.

Each environment writes one `best_model.keras` checkpoint. It atomically replaces an older checkpoint only after the current run records finite training and validation losses and produces a fresh checkpoint.

## Scope and assumptions

The paper does not publish its code, exact profile distributions, transformation pseudocode, seed, or capture assignment. It also trains its Random Forest on CICFlowMeter aggregate features, whereas this extension intentionally uses raw packet sequences so the CNN-BiGRU can learn spatial and temporal structure directly. Consequently, this changes both the classifier and its input representation; results should not be described as exact numerical reproduction of the paper.

The simulation treats application bytes as opaque, direction-specific capacity and does not carry real payloads. It therefore measures byte accounting rather than bit-exact decoding. Network loss, congestion, TCP retransmissions, synchronization metadata, and application-visible correctness require the excluded prototype/emulation phase.

When a source epoch exceeds one target template's capacity, the simulator represents the queued overflow as additional target-shaped windows before aggregating its prediction. This approximates the paper's next-epoch buffering without implementing the excluded live scheduler.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
