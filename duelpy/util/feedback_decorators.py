"""Decorators to alter the behavior of feedback mechanisms."""

from duelpy.feedback import FeedbackMechanism


class FeedbackMechanismDecorator(FeedbackMechanism):
    """A feedback mechanism that delegates to another feedback mechanism.

    This is intended  to be used as a base class for wrappers that want to
    "inject" some behavior or checks into an existing feedback mechanism. See
    ``BudgetedFeedbackMechanism`` for an example.

    Parameters
    ----------
    feedback_mechanism
        The FeedbackMechanism object to delegate to.
    """

    def __init__(self, feedback_mechanism: FeedbackMechanism) -> None:
        # We override all functions and delegate to an existing feedback
        # mechanism. Therefore it does not make much sense to call the super
        # constructor here.
        # pylint: disable=super-init-not-called
        self.feedback_mechanism = feedback_mechanism

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        return self.feedback_mechanism.duel(arm_i_index, arm_j_index)

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        return self.feedback_mechanism.get_num_duels()

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.feedback_mechanism.get_arms()

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return self.feedback_mechanism.get_num_arms()
