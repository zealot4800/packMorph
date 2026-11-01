# PackMorph

PackMorph supports two separate datasets, with independent VPN and NonVPN
classifiers. VNAT also supports six packet-derived obfuscation adapters: FlowShift,
BLANKET, Ditto, MiniPatch, Minos and Securitas.

| Dataset | Location | Services | Model inputs | Packet defenses |
| --- | --- | --- | --- | --- |
| VNAT | `dataset/VNAT/` | 5 | 20 packet-derived features per environment | Supported with saved packet context |
| CIC-ISCX | `dataset/CIC-ISCX/` | 7 | 23 timing features | Requires original packet data |

Each dataset contains `Non VPN/{train,val,test}.pkl` and
`VPN/{train,val,test}.pkl`. Their labels, features, sampling policies and evaluation
limits remain separate. They are not pooled or relabeled to force matching classes.

## Setup

Python 3.12 or 3.13 is required:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Select a dataset

Commands accept `--dataset vnat` or `--dataset cic-iscx`. Omitting the option
uses `default_dataset` in `configs/default.yaml` (currently `vnat`). Use
`--config path/to/config.yaml --dataset cic-iscx` to select a profile from a custom
shared configuration.

```bash
# Train separate VPN and NonVPN baseline models for each dataset.
.venv/bin/python -m scripts.train all --dataset vnat
.venv/bin/python -m scripts.train all --dataset cic-iscx

# Evaluate original test features with the matching frozen models.
.venv/bin/python -m scripts.evaluate baseline --dataset vnat
.venv/bin/python -m scripts.evaluate baseline --dataset cic-iscx
```

To train only one environment, replace `all` with `vpn` or `non-vpn`.
`./run_all.sh vnat` or `./run_all.sh cic-iscx` trains and evaluates the corresponding
baseline. Full training uses the episode count in the selected configuration.

Both datasets are configured in [configs/default.yaml](configs/default.yaml):

- `default_dataset`: profile used when `--dataset` is omitted.
- `model`: shared classifier/training settings.
- `obfuscation`: shared defense settings, used where packet data supports them.
- `datasets.vnat` and `datasets.cic-iscx`: dataset-specific output paths, data
  schemas, services and model feature lists. Optional `model` or `obfuscation`
  overrides inside a profile take precedence over shared values.

Results remain isolated:

```text
experiments/VNAT/results/frozen_baseline/cic-pkl-v2-config20/{non-vpn,vpn}/
experiments/CIC-ISCX/results/frozen_baseline/cic-pkl-v2-config23/{non-vpn,vpn}/
```

Manifests bind the source schema, labels, configured features, split identities,
training settings and capture-separation status. Historical artifacts under
`results/` are not automatically migrated or reused. Train models explicitly for
these configurations before evaluating or regenerating learned defenses.

## Training and overfitting controls

The shared `model` settings in `configs/default.yaml` govern both datasets.
`episodes` is a maximum, not a required duration. Validation is checked every
250 optimizer episodes. Training stops after eight checks without an improvement
larger than `min_delta` (0.001), and restores the checkpoint with the highest
observed validation score, including smaller improvements. Ties retain the earlier
checkpoint. Patience values count validation checks, not full passes over data.

The learning rate is multiplied by 0.5 after three stalled validation checks,
with a floor of 0.00001. Adam uses weight decay 0.0001 and global gradient clipping
at norm 1.0. Dropout remains 0.25 and can be configured. These are starting
settings, not a guarantee of higher accuracy.

Training uses unique original flows. Scaling is fitted on those unique training
rows, so oversampled PKL copies cannot change its statistics. Checkpoint scoring
uses capture/class macro accuracy for VNAT and class macro accuracy for CIC-ISCX.
Only validation controls stopping, learning-rate changes and checkpoint choice.
Test metrics are computed after selection; do not tune settings against them.

Each model artifact includes `training_history.json`, with episodic training loss,
training/validation macro scores and learning rate at every checkpoint. Training
scores include calibration-support examples and are diagnostic, not independent
holdout estimates. The manifest also reports `unique_training_metrics` separately
from the balanced PKL's training metrics.

New artifacts use `cic-pkl-v2-config<N>` directories. Existing v1 artifacts remain
intact and are not silently reused. Retrain using the same dataset commands above.
Learned defense caches tied to old classifier weights may require refitting with
`obfuscate ... --refit` after retraining.

VNAT validation sets are small (145 NonVPN flows and 90 VPN flows), so checkpoint
scores can be noisy. CIC-ISCX still lacks capture IDs. These training controls do
not remove either evaluation limitation.

## Dataset behavior

VNAT retains its five classes: Streaming, VoIP, Chat, C2 and File Transfer.
The loader enforces class balance, consistent training copies, disjoint original
row/record identities and capture separation. Training sampling and checkpoint
scoring weight captures equally within each class. Saved policies and packet
context remain in the original PKLs.

CIC-ISCX retains its seven classes: BROWSING, CHAT, STREAMING, MAIL, VOIP, P2P and
FT. The selected files are the prepared 15-second version. Natural class counts
are retained. The loader validates training-only median imputation, provenance and disjoint
flow identities. It reads `X`, never
`X_scaled`; the classifier fits its own log1p scaler on training inputs once.
Training balances services through sampling, while validation/test keep their
natural distributions. Checkpoint selection uses class macro accuracy.

CIC-ISCX has no capture IDs. Capture-balanced metrics are reported as unavailable,
and capture separation remains unverified. Its holdouts support exploratory
flow-level evaluation, not a claim of generalization to unseen captures. The code
does not invent capture IDs or packet sequences. See
[the CIC-ISCX guide](dataset/CIC-ISCX/README.md) for loading details.

The classifier supports 1–35 configured features and uses four 64-unit dense
layers followed by class-prototype scoring. Existing dataset PKLs are not
rewritten by loading or baseline training.

## Scripts

The scripts directory contains the current experiment commands:

| Script | Purpose |
| --- | --- |
| `train.py` | Train/load dataset-specific baseline models |
| `obfuscate.py` | Generate VNAT packet-defense features |
| `evaluate.py` | Evaluate baselines or saved defense features |
| `stream_flowshift.py` | Incremental service libraries and causal raw-PCAP replay |

The one-time Tor/NonTor and Extra raw-data preparation scripts have been removed.
Feature experiments use the prepared VNAT and CIC-ISCX PKLs. Streaming FlowShift
uses original PCAPs, with the VNAT PKLs supplying training/holdout capture lists.

## VNAT packet defenses

```bash
.venv/bin/python -m scripts.obfuscate minos --dataset vnat --all-flows
.venv/bin/python -m scripts.evaluate minos --dataset vnat
.venv/bin/python -m scripts.evaluate all --dataset vnat
```

Available defenses are `flowshift`, `blanket`, `ditto`, `minipatch`, `minos` and `securitas`.
Use `obfuscate all` to generate features for all six. Learned defenses can require
matching frozen baseline models; train them explicitly before generation.

Defense generation rejects CIC-ISCX before fitting policies or writing caches:
aggregate timing features cannot reconstruct the packet `flow_context` required
by these adapters. Original PCAP data and a suitable feature adapter are needed
to extend support.

VNAT uses the same 20 user-selected packet-derived features in both environments. The stored feature order
is part of the dataset and baseline schema.

For continuous FlowShift, use the raw-packet simulator:

```bash
# Scapy's file reader is available through the optional traces dependency.
.venv/bin/pip install -e '.[traces]'

# Learn from up to 3 training captures per service, 2,000 packets each.
# The default raw-data root is /home/zealot/NDataset/Data/Encrypted-Traffic-Datasets/VNAT.
.venv/bin/python -m scripts.stream_flowshift build --environment non-vpn \
  --library experiments/VNAT/results/flowshift/online/non-vpn-library.json

# Replay held-out validation capture prefixes; library stays frozen.
.venv/bin/python -m scripts.stream_flowshift replay --environment non-vpn \
  --library experiments/VNAT/results/flowshift/online/non-vpn-library.json \
  --output experiments/VNAT/results/flowshift/online/non-vpn-validation
```

Repeat with `--environment vpn` and separate library/output paths for VPN.
`--traces` overrides the raw-data root. `build --update` adds unseen training
captures within the saved library bounds; it does not reread existing prefixes.

`ServiceLibrary.observe(service, capture_id, flow_id, packet)` incrementally
records real labeled reference packets, with bounded samples per capture and
bounded flow clocks. Candidate policies retain separate capture templates so a
long capture cannot receive extra weight simply by contributing more packets.
There is no classifier training in this path.
`OnlineFlowShift(library, source_service).push(StreamPacket(time, direction, data))`
transforms each arrival and returns committed release events. Each epoch compares
source-to-target strengths using the recent observed prefix, retains the Pareto
frontier for target progress/bytes/buffering, and chooses a positive cost-adjusted
score. With no useful candidate it uses minimum-cost tunnel framing. Reference
updates change the next epoch's candidates; an active policy retains its snapshot.
Only the original datagrams already received earn shaping-byte credit. Fragment
headers, padding and dummies spend that credit, and release times preserve each
direction's order and obey `--max-delay`. Idle flows emit no unbounded cover traffic.

To exercise **online library updates during replay**, add
`--online-reference-packets 100` with a single `--capture <held-out.pcap>` and
`--source-service <service>`. Each saved training feed resumes after its verified
prefix, aligned to simulation time zero at that boundary. Only due reference
events update the library; held-out packets never do. The updated library is
saved as `library-updated.json` for reuse. This alignment models independent
reference feeds, not the captures' original calendar chronology.

Replay writes ordered packet events and a report, checking byte-exact restoration
of every datagram. Defaults are 1-second epochs, a 0.25% cumulative shaping-byte
budget, and 20 ms maximum buffering. Total wire cost also includes modeled
IPv4/UDP headers (28 bytes), restoration records (16), authentication tags (16),
and mandatory MTU fragmentation. `--shaping-budget` caps additional shaping,
not this mandatory tunnel cost. Backward capture timestamps up to 5 microseconds
are clamped to the current clock and counted; `--timestamp-tolerance 0` rejects
all reversals. Neither encryption, link capacity, congestion nor network FCT is
simulated. Target-profile progress is a selection heuristic: classifier accuracy
must be measured from replayed traffic before claiming superiority. These prefix
replays do not establish accuracy gains or full-capture generalization.

The existing **aggregate FlowShift adapter** remains available for comparison.
Its default guide is fitted to the matching
frozen classifier using unique training flows and all their feasible edits.
Sampling balances services and captures, then flows and candidate edits. A
classification-and-margin loss fits the guide; validation policy accuracy and
actual byte/span costs select the checkpoint, with early stopping and learning
rate reductions. Validation never enters gradient updates. Generation ranks
edits without target-classifier queries or service labels.

`cost_basis: aggregate` prices added bytes against a fixed training-derived
reference so large flows cannot dominate aggregate overhead cheaply. The tuner
returns non-dominated policy artifacts and a ready-to-use configuration. By
default, it selects a measured improvement over a compatible incumbent without
increasing its validation accuracy, bytes or mean span; `--allow-tradeoff` permits
another operating point. The default validation aggregate byte cap is 0.25%.
This selection cap does not replace the hard per-flow limits or guarantee the
same aggregate cost on new traffic. Each environment has just one VoIP training
capture, which limits conclusions about capture generalization.

```bash
.venv/bin/python -m scripts.obfuscate flowshift --dataset vnat
.venv/bin/python -m scripts.evaluate flowshift --dataset vnat

# Train and select FlowShift policies on validation; test predictions are unused.
.venv/bin/python -m scripts.tune_flowshift --dataset vnat

# Force a fresh fit, or set a different validation byte cap.
.venv/bin/python -m scripts.tune_flowshift --dataset vnat --refit --max-aggregate-bytes 0.0025
```

Tuning saves checkpoint history, policy costs, a Pareto policy family and selected
artifacts under `experiments/VNAT/results/flowshift/`. It leaves dataset PKLs and
the source config unchanged. Use its printed `selected-config-*.yaml` with
`--config` (omit `--dataset` for that resolved config). To reuse the exact selected
weights, pass `--flowshift-policy <selected.pkl>` and a single `--environment`
to `scripts.obfuscate`; use `--split val` to generate validation features.
