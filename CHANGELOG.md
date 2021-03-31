# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The feedback mechanism is now automatically wrapped with a
  `BudgetedFeedbackMechanism` by the `Algorithm` superclass. This reduces the
  need for algorithm-specific early termination logic. The `step` function may
  now raise an exception because of this wrapper. The exception is caught in
  `run`. This is a breaking change.
- `BudgetedFeedbackMechanism` now throws an object-local exception. That makes
  it possible to only catch exactly the intended exception.
- `BudgetedFeedbackMechanism` now accepts `max_duels=None` for an unlimited
  budget. That can be useful in some situations when the budget should be
  applied conditionally.
- The duel count was removed from the main `FeedbackMechanism` class. You can
  use a `BudgetedFeedbackMechanism` (accessible on all `Algorithm` instances by
  `instance.wrapped_feedback` and its `duels_conducted` attribute as a
  replacement. As a result the `FeedbackMechanism` superclass no longer
  requires any state. Subclasses may or may not have state. For example
  `MatrixFeedback` still requires a numpy random state.

### Fixed

- The `MergeRUCB` implementation now explores until only one arm remains and
  then continues with exploitation. The stage counter is incremented less
  often, which affects the merge frequency.

## [0.1.0] - 2021-03-04

### Added

- Algorithm implementations (see the documentation for references).
	- ApproximateProbability
	- BeatTheMeanBandit
	- BeatTheMeanBanditPAC
	- CopelandConfidenceBound
	- DoubleThompsonSampling
	- DoubleThompsonSamplingPlus
	- InterleavedFiltering
	- KLDivergenceBasedPAC
	- KnockoutTournament
	- MallowsMPI
	- MallowsMPR
	- MergeRUCB
	- Multisort
	- OptMax
	- PlackettLuceAMPR
	- PlackettLucePACItem
	- RelativeConfidenceSampling
	- RelativeUCB
	- Savage
	- ScalableCopelandBandits
	- SequentialElimination
	- SingleEliminationTop1Select
	- SingleEliminationTopKSorting
	- SuccessiveElimination
	- WinnerStaysStrongRegret
	- WinnerStaysWeakRegret
- Utilities for implementing PB-MAB algorithms.
	- A class for preference estimations.
	- Superclasses for algorithm implementations.
	- And more. See the documentation for details.
- A basic experiment runner.
- Documentation.

## [0.0.1] - 2020-06-03

### Added

- A basic skeleton of the project without any functionality.

[Unreleased]: https://gitlab.com/duelpy/duelpy/compare/v0.1.0...master
[0.1.0]: https://gitlab.com/duelpy/duelpy/compare/v0.0.1...v0.1.0
[0.0.1]: https://gitlab.com/duelpy/duelpy/-/releases/v0.0.1

