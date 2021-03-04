# duelpy

This is a python package for solving with Preference Based Multi Armed Bandit problems, also known as dueling bandits. Refer to [this paper](https://jmlr.org/papers/v22/18-546.html) for an overview of the field.

You can compare the implemented algorithms in an experiment by running

```
python3 -m duelpy.experiments.cli
```

The experiments are still rather limited. The command-line interface can
currently only run regret-based comparisons in a limited set of configurations.
Pass the `--help` flag for more information.

See [the documentation](https://duelpy.gitlab.io/duelpy/) for more information about the implemented algorithms.
