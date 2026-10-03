# Scenario evaluation summary

Stream windows = windows after the 14 reference windows. Detection counts as on time if the first alarm comes within 3 windows of the drift end (gradual) or drift start (sudden).

| Scenario | Detector | Expected | Stream alarms | False alarms before change | First alarm (delay) | Alarm rate after change |
|---|---|---|---|---|---|---|
| natural | driftlens | unknown (real) | 122/139 | - | - | - |
| natural | mcddd | unknown (real) | 2/139 | - | - | - |
| sudden | driftlens | drift at 30 | 30/46 | 0 (0%) | 30 (+0) | 100% |
| sudden | mcddd | drift at 30 | 0/46 | 0 (0%) | none | 0% |
| gradual | driftlens | drift at 30-45 | 28/46 | 0 (0%) | 32 (+2) | 93% |
| gradual | mcddd | drift at 30-45 | 0/46 | 0 (0%) | none | 0% |
| volume | driftlens | no drift (volume x3 at 30) | 0/46 | 0% | - | 0% (any alarm is false) |
| volume | mcddd | no drift (volume x3 at 30) | 0/46 | 0% | - | 0% (any alarm is false) |
| no_drift | driftlens | no drift | 0/46 | - | - | - |
| no_drift | mcddd | no drift | 0/46 | - | - | - |
