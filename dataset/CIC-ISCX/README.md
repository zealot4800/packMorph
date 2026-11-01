# CIC-ISCX 15-second service dataset

The six PKLs are the prepared 15-second, seven-service dataset:
`Non VPN/{train,val,test}.pkl` and `VPN/{train,val,test}.pkl`.

| Environment | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Non VPN | 6013 | 1289 | 1288 |
| VPN | 6481 | 1389 | 1387 |

Service IDs follow `class_names`: BROWSING, CHAT, STREAMING, MAIL, VOIP, P2P, FT.
These classes differ from VNAT's five classes; do not combine label IDs.

```bash
.venv/bin/python -m scripts.train all --dataset cic-iscx
.venv/bin/python -m scripts.evaluate baseline --dataset cic-iscx
```

`X` contains only the 23 median-imputed original timing features,
in the order recorded in `feature_names`. Imputation used training medians only.
`X_raw` and `missing_mask` preserve missingness as metadata; missing-value
indicators are excluded from model inputs. The shared loader checks `X`
against those training statistics and checks record/provenance overlap. Labels,
source lines and filenames are metadata, not model inputs.

The baseline reads `X`, fits its own training-only log1p standardization and
reuses that scaler for validation/test. The separately saved `X_scaled` is useful
for external classifiers but is not used by PackMorph: passing it through the
baseline scaler again would double-transform the features.

Class counts remain natural. Episodic training samples services equally and flows
uniformly within each service. Validation checkpoint selection uses class macro
accuracy. There are no real capture IDs: capture metrics are null, not estimates.
The stratified flow-level splits cannot establish unseen-capture generalization.
Packet-based obfuscation is unavailable without the original PCAP/context data.

The raw ARFF source files and the original preprocessing/audit artifacts are not
present in the current dataset directory. Source filenames, line numbers and the
source fingerprint remain inside each PKL. The one-time raw-data preparation scripts have been retired; this checkout
uses the prepared PKLs. Rebuilding from ARFF would require restoring the original
preprocessing tooling and source data. Do not replace these splits casually or mix
in other timeout variants, which contain overlapping source traffic.

Settings are in `datasets.cic-iscx` in `configs/default.yaml`, alongside the
shared training settings and VNAT profile. Results are under
`experiments/CIC-ISCX/results/`, separate from VNAT.
