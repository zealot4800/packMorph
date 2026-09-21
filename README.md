# FlowShift

Minimal implementation of the architecture described in `TMS_Journal.pdf`.
FlowShift runs as a user-space shim between an application and TCP. The sender
changes observable size, volume, and timing characteristics; the receiver
reverses those changes and reconstructs the original file.

Implemented operations:

- fragmentation and padding;
- dummy-frame insertion;
- inter-frame timing;
- seeded policy selection;
- receiver-side restoration;
- SHA-256 correctness verification;

## Policies and pseudorandom schedule

Each file in `configs/` represents one complete experiment and contains its
own ordered policy bank. Every policy supplies the three transformation
dimensions from the paper:

```text
target_sizes             S: transformed frame sizes
inter_packet_ms          T: delays between released frames
target_frames_per_epoch  V: real-plus-dummy frame volume
```

File-transfer configurations use combinations of interactive, voice,
streaming, and bulk target patterns. Streaming configurations use low-latency,
buffered, and burst-video patterns.

For interval `k`, both endpoints independently calculate:

```text
(u_k, v_k) = HMAC-SHA256(shared_seed, k)
policy_index = u_k mod number_of_policies
delta_time = delta_min + v_k * (delta_max - delta_min)
```

`u_k` selects a policy from the ordered list. `v_k` is in `[0, 1)` and selects
how long that policy remains active. Nothing has to signal a new policy for
every interval because the sender and receiver have the same configuration,
seed, and interval number.

Inspect the next generated selections before an experiment:

```bash
./flowshift schedule \
  --config configs/file-transfer-500mb.json \
  --count 10
```

Changing the seed creates a different schedule. The sender includes the chosen
configuration in the flow setup metadata so the receiver can reproduce it.

## Repository

```text
configs/file-transfer-500mb.json
configs/file-transfer-1gb.json
configs/file-transfer-2gb.json
configs/streaming-60s-720p.json
configs/streaming-120s-720p.json
configs/streaming-180s-720p.json
flowshift                         repository-local launcher
run-vms                           host-side two-VM launcher
src/flowshift/             sender, receiver, protocol and planner
TMS_Journal.pdf            design paper
```

## VirtualBox network

Use a separate Internal Network adapter for the experiment:

```text
VM1: 10.10.0.1/24
VM2: 10.10.0.2/24
```

Keep a different NAT or Host-Only adapter for management and copying the
repository. Replace `enp0s9` below with the experimental interface name shown
by `ip -br link`.

VM1:

```bash
sudo ip address add 10.10.0.1/24 dev enp0s9
sudo ip link set enp0s9 up
```

VM2:

```bash
sudo ip address add 10.10.0.2/24 dev enp0s9
sudo ip link set enp0s9 up
```

Do not add a default route through the experimental interface.

## Requirements

Ubuntu's system `python3` is the only FlowShift runtime requirement. Actual
video streaming additionally requires the Ubuntu `ffmpeg` package on both VMs.

## Keep one copy on the host

The repository does not need to be copied into either VM. In VirtualBox, add
the current host directory to both VMs as the same shared folder:

```text
Host path:  /home/zealot/ResarchWork/VM_Flowshift
Share name: flowshift
Read-only:  enabled
```

VirtualBox Guest Additions must be available in both Ubuntu guests. Mount the
share in each VM:

```bash
sudo mkdir -p /mnt/flowshift
sudo mount -t vboxsf -o ro flowshift /mnt/flowshift
```

The source remains in one place on the host. VM1 and VM2 only execute the
mounted files. Store received files outside the read-only shared directory.

## File-transfer experiments

The file-transfer configuration provides exact decimal sizes:

```text
500mb =   500,000,000 bytes
1gb   = 1,000,000,000 bytes
2gb   = 2,000,000,000 bytes
```

Create sparse experiment inputs from the repository root. They have the exact
logical sizes but consume little host storage until written:

```bash
mkdir -p workloads
truncate -s 500000000 workloads/file-500mb.bin
truncate -s 1000000000 workloads/file-1gb.bin
truncate -s 2000000000 workloads/file-2gb.bin
```

Run a selected experiment entirely from the host:

```bash
./run-vms --file 500mb /mnt/flowshift/workloads/file-500mb.bin
./run-vms --file 1gb /mnt/flowshift/workloads/file-1gb.bin
./run-vms --file 2gb /mnt/flowshift/workloads/file-2gb.bin
```

The sender validates the input size against the selected experiment before
transmission. Sender and receiver SHA-256 values must match.

## Stream and play an actual video

The streaming configuration provides `60s`, `120s`, and `180s` experiments.
Every profile encodes at 1280×720, 1200 kbps video, and 96 kbps audio. Place an
actual video in `workloads/` so VM1 can access it through the shared folder.
Both VMs need FFmpeg:

```bash
sudo apt-get install -y ffmpeg
```

Keep VM2 logged into its Ubuntu desktop, then run this command on the host:

```bash
./run-vms --stream 60s /mnt/flowshift/workloads/video.mp4
./run-vms --stream 120s /mnt/flowshift/workloads/video.mp4
./run-vms --stream 180s /mnt/flowshift/workloads/video.mp4
```

VM1 loops a short source when necessary and stops at the configured duration.
FFmpeg applies real-time input pacing and produces a 720p low-latency MPEG-TS
stream. FlowShift transforms the live stream. VM2 restores it and pipes it into
FFplay while bytes arrive, so playback does not wait for the complete video.

The restored MPEG-TS stream is also saved as `~/received/video.ts` on VM2 and
verified using SHA-256. If the Python policy executor cannot sustain the 1296
kbps combined media rate, playback will buffer and completion can extend past
the nominal experiment duration; this is a measurable result, not hidden by
the launcher.

## Capture transformed traffic

On VM2, find the experimental interface and capture the transfer:

```bash
sudo tcpdump -i enp0s9 -s 0 -w flowshift.pcap tcp port 9443
```

TCP segmentation, retransmissions, and virtual-NIC offloading can
change packet sizes observed in the PCAP. Use packet captures—not application
`send()` sizes—when evaluating the transformation.

The provided policies are controlled experiment profiles. Paper-quality
classification claims still require profiles learned from representative,
capture-disjoint traffic data.
