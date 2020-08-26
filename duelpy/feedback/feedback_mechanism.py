"""A generic way to compare two arms against each other."""


class FeedbackMechanism:
    """Some means of comparing two arms."""

    def __init__(self, list_arm_labels: list) -> None:
        self.list_arm_labels = list_arm_labels

    # In our final design we will probably want a better arm representation to
    # avoid restricting it to int.
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
        raise NotImplementedError

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.list_arm_labels

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return len(self.list_arm_labels)
