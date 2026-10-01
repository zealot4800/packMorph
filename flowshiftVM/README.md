# FlowShift VM workloads and derived policies

The six JSON files in `configs/` describe 500 MB, 1 GB, and 2 GB file
transfers, plus 60, 120, and 180 second 720p streams. Their policy sizes and
frame counts are generated from the dataset's CICFlowMeter CSVs. Regenerate
them after changing the dataset:

```bash
python3 flowshiftVM/derive-configs
```

The default calibration uses `dataset/CSV/VPN/{File Transfer,Streaming}` and a
250 Mbps link. To calibrate against the non-VPN captures or another link:

```bash
python3 flowshiftVM/derive-configs --environment 'Non VPN' --link-bps 100000000
```

The command selects 80% of eligible capture CSVs by a stable hash of the
capture path and calibration seed (default 42), leaving the rest out. A flow
is eligible when it has at least 32 packets, lasts at least 0.1 seconds, and
has finite packet-length and byte-rate summaries. It takes the 10th, 30th,
70th, and 90th percentile of eligible **flow means** in each capture, then
the median of each percentile across captures. Thus a capture containing many
flows does not dominate the profile. The generated `derivation` block records
counts, quantiles, rates, and a hash of the selected input files; the VM
needs no capture file paths at runtime.

File-transfer policies move packet sizes by 25%, 50%, 75%, or 100% from the
file-transfer profile toward the streaming profile. Streaming policies move
in the opposite direction. These values are **size-profile estimates** from
CICFlowMeter, not observed VM application payload sizes. Each policy's frame
slots per 250 ms epoch are computed from the expected fragments per 64 KiB
file read or 32 KiB media read, with 2% slot headroom. File-transfer input
rate is the median of capture-level median byte rates, limited by link
capacity after estimated VM framing overhead. Streaming input rate comes
from the configured video and audio bitrates. The `40` byte VM frame header
is included in the link estimate. The generated PRG seed is deterministic
for reproducibility and is not a shared secret.

The epoch is fixed at 250 ms because the current VM sender uses a fixed
frame-slot count per policy. File-transfer per-packet delays are zero to avoid
thousands of Python sleeps per epoch; streaming uses half the epoch for
configured per-packet delays and leaves the rest for I/O. The fixed epoch,
slot headroom, and media bitrate are experiment controls. The streaming delay
is capped at the smaller of its observed median flow IAT and half an epoch
divided by its frame slots; this cap may be much smaller than the observed
IAT to meet the media rate. These configured delays are therefore a pacing
proxy, not a replay of the capture timestamps. The current sender chooses
among every policy in the JSON using its
shared PRG; it does not run the simulator's Pareto selector. The CSV estimates
also cannot predict socket scheduling, congestion, retransmissions, or the
actual video encoder's output rate. Measure byte overhead and completion time
from a VM run before making performance claims.

Check the generated configs without starting the VMs:

```bash
for config in flowshiftVM/configs/*.json; do
  flowshiftVM/flowshift schedule --config "$config" --count 1 >/dev/null
done
```

Use `flowshiftVM/create-workload` to build a workload and `flowshiftVM/run-vms`
to execute it. The three file sizes and three stream durations share one
policy family per workload type; their sizes or durations change the run
length, not the observed service-class profile.
