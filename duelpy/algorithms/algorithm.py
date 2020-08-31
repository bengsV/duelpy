"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""


class Algorithm:
    """Parent class of all the implemented PB-MAB algorithms."""

    def step(self) -> None:
        """Run one round of the algorithm."""
        raise NotImplementedError

    def run(self, rounds: int) -> None:
        """Run the algorithm for a given number of rounds.

        Parameters
        ----------
        rounds
            The number of rounds for which the algorithm is run.
        """
        for _ in range(rounds):
            self.step()
