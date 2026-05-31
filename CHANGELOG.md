# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Adversarial dueling bandit support (Saha, Koren & Mansour, 2021).
	- `DuelingExp3` and `DuelingExp3HighProbability` algorithms (Algorithms 1
	  and 2): EXP3-style regret minimization against the Borda winner.
	- `BordaConfidenceBound` algorithm (Algorithm 3): an explore-then-commit
	  algorithm for the fixed-gap adversarial setting.
	- `AdversarialMatrixFeedback`: a feedback mechanism driven by a time-varying
	  sequence of preference matrices.
	- `FixedGapAdversarialMatrix`: a stationary fixed-gap Borda environment.
	- `BordaRegret`: a metric measuring regret against the Borda winner.

### Changed

Nothing yet.

### Fixed

Nothing yet.

## [1.0.0] - 2021-03-31

### Added

- Algorithm implementations (see the documentation for references).
	- ActiveRanking
	- BinarySearchRanking
	- BordaRanking
	- CwRmed
	- Rmed1
	- Rmed2
	- Rmed2FH
	- VerificationBasedCondorcet
- Algorithm result superclasses
	- BordaRankingProducer
	- GeneralizedRankingProducer
- A new `duel_repeatedly` function that can be implemented for a
  `FeedbackMechanism` to optimize repeated duels of the same arms. A custom
  implementation is optional. The default implementation falls back to repeated
  calls of `duel`.

### Changed

- The feedback mechanism is now automatically wrapped with a
  `BudgetedFeedbackMechanism` by the `Algorithm` superclass. This reduces the
  need for algorithm-specific early termination logic. The `step` function may
  now raise an exception because of this wrapper. The exception is caught in
  `run`. This is a breaking change.
- As a consequence of the last bullet point the `feedback_mechanism` attribute
  has been replaced by `wrapped_feedback`. This should make it clear that the
  feedback mechanism is wrapped and avoid surprises.
- `BudgetedFeedbackMechanism` now throws an object-local exception. That makes
  it possible to only catch exactly the intended exception.
- `BudgetedFeedbackMechanism` now accepts `max_duels=None` for an unlimited
  budget. That can be useful in some situations when the budget should be
  applied conditionally.
- The decision example (`examples/decision.py`) now uses a feedback mechanism
  decorator to implement the printing of the current preference estimate. This
  is because `step` does not execute just a single duel in some of our
  algorithm implementations. It may also raise an exception now. It is now best
  practice to use decorators instead of relying on calling `step` manually.
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
- The relative preference computation in `PlackettLuceModel` has changed. The
  previous calculation was wrong due to operator precedence.
- The `WinnerStaysWeakRegret` and `WinnerStaysStrongRegret` implementations now
  inherit from `CondorcetProducer`.
- The threshold computation in `SuccessiveElimination` was fixed. Operator
  precedence strikes again.
- `SingleEliminationTopKSorting` would previously spend its entire duel budget
  on a single execution of `SingleEliminationTop1Select` (including
  exploitation). This has been fixed.
- `MallowsMPI` in "PAC mode" (no time horizon) would previously terminate as
  soon as only two arms remain, without properly determining the winner of the
  two. This has been fixed.
- The `KLDivergenceBasedPAC` component of `ScalableCopelandBandits` no longer
  terminates exploration during reward generation in "PAC mode".
- The number of comparisons in `SingleEliminationTop1Select` and
  `SingleEliminationTopKSorting` was previously computed with a base-2
  logarithm although a natural logarithm should be used. This has been fixed.

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

[Unreleased]: https://gitlab.com/duelpy/duelpy/compare/v1.0.0...master
[1.0.0]: https://gitlab.com/duelpy/duelpy/compare/v0.1.0...v1.0.0
[0.1.0]: https://gitlab.com/duelpy/duelpy/compare/v0.0.1...v0.1.0
[0.0.1]: https://gitlab.com/duelpy/duelpy/-/releases/v0.0.1

