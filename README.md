# duelpy

This is a python package for solving with Preference Based Multi Armed Bandit problems, also known as dueling bandits. Refer to [this paper](https://arxiv.org/abs/1807.11398) for an overview of the field.

You can compare the implemented algorithms in an experiment by running

```
python3 duelpy.experiments.savage_experiment
```

The experiments are still a work in progress, for now only a single experiment
is implemented. The experiment compares average regret and algorithm runtime
with a fixed time horizon. Pass the `--help` flag for more information.
