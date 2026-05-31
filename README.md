# duelpy

A Python library for **Preference-Based Multi-Armed Bandit** (PB-MAB) problems, also known as dueling bandits. Refer to [this paper](https://jmlr.org/papers/v22/18-546.html) for an overview of the field.

## Installation

```bash
pip install duelpy
```

## Quick start

```python
import numpy as np
from duelpy.feedback import MatrixFeedback
from duelpy.algorithms import RelativeUCB

# Define a 3-arm preference matrix where p[i,j] = P(arm i beats arm j)
preference_matrix = np.array([
    [0.5, 0.7, 0.8],
    [0.3, 0.5, 0.6],
    [0.2, 0.4, 0.5],
])

feedback = MatrixFeedback(preference_matrix, random_state=np.random.RandomState(42))
algorithm = RelativeUCB(feedback, time_horizon=500)
algorithm.run()
print("Best arm:", algorithm.get_condorcet_winner())
```

## Running experiments

You can compare multiple algorithms through the built-in CLI:

```bash
python3 -m duelpy.experiments.cli --help
```

## Development setup

```bash
git clone <repo-url>
cd duelpy
pip install -e .
pip install pre-commit
pre-commit install
pytest
```

## Documentation

See [the full documentation](https://bengsv.github.io/duelpy/) for the complete API reference and algorithm descriptions.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
