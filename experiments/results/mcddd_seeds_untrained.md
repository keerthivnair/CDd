# MCD-DD across model seeds [42, 1, 2, 3, 4]

Scenario config MCD-DD settings with overrides {"train_epochs": 0}.

Each cell: median over seeds [min-max]. Only MCD-DD is refit; embeddings and scenarios are fixed.

| Scenario | Stream alarms | False alarms before change | First-alarm delay | Detected (runs) |
|---|---|---|---|---|
| sudden | 12 [10-12] | 0 [0-0] | 0 [0-0] | 5/5 |
| gradual | 15 [10-16] | 0 [0-1] | 6 [4-6] | 5/5 |
| volume | 0 [0-2] | 0 [0-2] (all false) | - | - |
| no_drift | 1 [0-2] | 1 [0-2] (all false) | - | - |
