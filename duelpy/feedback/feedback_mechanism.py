"""A generic way to compare two arms against each other."""


class FeedbackMechanism:
    """Some means of comparing two arms."""

    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms

    # In our final design we will probably want a better arm representation to
    # avoid restricting it to int.
    def duel(self, arm_i: int, arm_j: int) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i
            The challenger arm.
        arm_j
            The arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        raise NotImplementedError

    def get_num_arms(self) -> int:
        """Get the number of arms available."""
        return self.num_arms
