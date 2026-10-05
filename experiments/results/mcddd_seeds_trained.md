# MCD-DD across model seeds [42, 1, 2, 3, 4]

Scenario config MCD-DD settings with overrides {"train_epochs": 100}.

Each cell: median over seeds [min-max]. Only MCD-DD is refit; embeddings and scenarios are fixed.

| Scenario | Stream alarms | False alarms before change | First-alarm delay | Detected (runs) |
|---|---|---|---|---|
| sudden | 9 [1-10] | 0 [0-1] | 1 [0-2] | 5/5 |
| gradual | 3 [1-8] | 1 [0-3] | 9 [6-14] | 4/5 |
| volume | 0 [0-0] | 0 [0-0] (all false) | - | - |
| no_drift | 0 [0-1] | 0 [0-1] (all false) | - | - |
