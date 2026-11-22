# Step-5 metric calibration (2026-11-22, `calibrate.sh`)

Reference is the full edit (S=1) on the prefill path (slice 64): D_A = on_s64 - off_s64.

| path B | result |
|---|---|
| full edit, decode path (slice 1), from gpu_validate.sh | norm ratio 0.992 | corr 0.925 | edit KL A 0.0109 B 0.0097 |
| S=0.9, same prefill path | norm ratio 0.925 | corr 0.911 | edit KL A 0.0109 B 0.0091 |
| S=0.9, decode path | norm ratio 0.922 | corr 0.877 | edit KL A 0.0109 B 0.0094 |
| S=0.5, same prefill path | norm ratio 0.694 | corr 0.740 | edit KL A 0.0109 B 0.0054 |
| embedding site broken, prefill path | norm ratio 1.032 | corr 0.830 | edit KL A 0.0109 B 0.0108 |
| embedding site broken on the decode path only | norm ratio 1.015 | corr 0.781 | edit KL A 0.0109 B 0.0103 |
