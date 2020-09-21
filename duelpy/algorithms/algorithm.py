"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""

from duelpy.feedback import FeedbackMechanism


class Algorithm:
    """Parent class of all the implemented PB-MAB algorithms."""

    def __init__(self, feedback_mechanism: FeedbackMechanism):
        self.feedback_mechanism = feedback_mechanism

    def step(self) -> None:
        """Run one step of the algorithm.

        This corresponds to a logical step of the algorithm and may perform
        multiple comparisons. What exactly a "logical step" is depends on the
        algorithm.
        """
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
