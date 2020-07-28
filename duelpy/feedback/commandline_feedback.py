"""Gather feedback from a human on the commandline."""

from duelpy.feedback.feedback_mechanism import FeedbackMechanism


class CommandlineFeedback(FeedbackMechanism):
    """Compare two arms based on human feedback on the CLI."""

    def duel(self, arm_i: int, arm_j: int) -> bool:
        """Perform a duel between two arms based on human feedback.

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
        print(f"Do you prefer arm {arm_i} (i) or arm {arm_j} (j)?")
        result = input("[i/j] ")
        while result not in {"i", "j"}:
            print('Please choose one of "i" or "j"')
            result = input("[i/j] ")
        arm_i_wins = result == "i"
        return arm_i_wins
