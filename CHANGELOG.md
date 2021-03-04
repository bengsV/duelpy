# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://gitlab.com/duelpy/duelpy/compare/v0.0.1...master
[0.0.1]: https://gitlab.com/duelpy/duelpy/-/releases/v0.0.1

