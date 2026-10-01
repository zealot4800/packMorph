# PackMorph simulation study

The study extracts packet metadata from the PCAP/PCAPNG files in `dataset/`,
compares five defenses, and writes `results/comparison_policy_bank.csv`. Install the Python
package from this directory and make sure `tshark` is available:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/packmorph-sim --config configs/default.yaml
```

## Offline FlowShift policy bank

For each environment and seed, `FlowShiftLibrary.fit` estimates size, timing,
and packet-count profiles from the **training captures only**. It combines
every source class with each other target class and every entry in
`methods.flowshift_manuscript.policy_variants` in `configs/default.yaml`.
The default five-class configuration has twelve variants, giving 48 prepared
candidate policies per source class. The bank is built before validation or
test flows are transformed.

Each variant independently controls movement toward the target class's size,
timing, and volume profile. Add or remove named entries in `policy_variants`
to change the bank. Larger banks take longer to evaluate. For each flow, the
simulator selects a non-dominated set using displacement and overhead
estimates from the training-only source profile, conditioned on the first
observed packet's direction and a 128-byte size bucket (capped at 2048 bytes).
It does not inspect later flow packets during selection. These are proxy
estimates, not measured costs of the unseen flow. It then applies the schedule to the complete
flow, checks byte conservation and realized overhead, and falls back to the
original trace when no policy is feasible or the combined schedule exceeds a
limit.

Extraction records a stable flow identity and each window's offset from the
flow start. The comparison runner joins windows from the same capture flow,
selects one policy set for the complete trace, and splits the transformed
trace back into windows for the CNN. Policy indices and holding times come
from a counter-based HMAC-SHA256 generator with a per-flow seed, so the same
seed and ordered selected list reproduce the schedule.

The comparison CSV contains one compact row per method, point, adversary,
environment, and seed. It keeps accuracy, macro F1, byte and packet overhead,
mean duration overhead, budget violations, incomplete records, fallbacks, and
deadline misses. The study settings and policy variants live in
`configs/default.yaml`; per-flow deadline outcomes have their own QoS CSV.
An older wide comparison CSV can be converted after its run finishes, without
retraining:

```bash
.venv/bin/packmorph-sim --config configs/default.yaml --compact-existing
```

This is a packet-trace simulation; receiver-side
restoration of actual application bytes from the manuscript cannot be tested
from PCAP metadata alone. The simulation checks that each original packet's
byte capacity and order survive fragmentation and dummy insertion. The separate
`flowshiftVM/` prototype still loads its own JSON policies; this simulation
bank is not exported to that prototype. The VM JSON policy parameters can be
regenerated from capture-flow CSV summaries with `python3 flowshiftVM/derive-configs`;
see `flowshiftVM/README.md` for its data split and modeling assumptions.

## Simulation QoS contracts

The optional `qos` section in `configs/default.yaml` sets a capture-local link
rate and supplies deadlines. It has no deadlines by default: PCAPs record
observed completion, not application requirements. For an explicit study
scenario, set a class default:

```yaml
qos:
  link_capacity_bps: 250000000
  class_defaults:
    File Transfer: {deadline_seconds: 30, maximum_bandwidth_overhead: 0.20}
  flow_contracts_csv: null
```

For individual flows, export the extracted flow IDs, then fill the service
requirements in the CSV:

```bash
.venv/bin/packmorph-sim --config configs/default.yaml --export-qos-template results/qos_contracts.csv
```

The export scans the configured captures and creates a new file; it will not
overwrite an existing contract. Set `flow_contracts_csv: ../results/qos_contracts.csv`
in `configs/default.yaml` after editing the rows. Required columns are:

```csv
environment,capture,flow_group,deadline_seconds,expected_trace_bytes,maximum_bandwidth_overhead,maximum_duration_overhead
```

`capture` is relative to `dataset/`; `flow_group` is the identity produced by
extraction. Blank values inherit class defaults; a row with every contract
field blank has no per-flow override. `expected_trace_bytes` is an optional
estimate known at flow start. If absent, the simulator uses the training
flows' 75th-percentile byte count. Unknown flow keys cause an error.

For each environment and seed, the simulator fits baseline duration and byte
estimates from training flows only. FlowShift rejects candidates predicted to
miss a contract deadline. During replay it can choose a faster feasible
policy or bypass later packets when estimated remaining work exceeds the
remaining time. It does not use unseen source packets for this decision.

Outcomes use one first-come-first-served bidirectional link per capture,
including frame serialization and queueing. The comparison CSV records
deadline miss counts; contracted-flow outcomes are saved to
`results/comparison_policy_bank.qos.<run_id>.csv`. Baseline misses and
defense-induced misses are reported separately.

Byte counts and `trace_goodput_bytes_per_second` use captured frame lengths
as a proxy. Encrypted PCAP metadata does not reveal application payload
bytes. The link model does not predict loss, retransmission, or VM processing.
Streaming playback deadlines require application segment boundaries and
playback times, which these PCAPs do not provide.
