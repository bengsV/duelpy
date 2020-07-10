"""A generic way to compare two arms against each other."""


class FeedbackMechanism:
    """Some means of comparing two arms."""

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
