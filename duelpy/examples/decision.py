"""A simple example of how a decision-tool using duelpy could work."""

from duelpy.algorithms import savage
from duelpy.feedback import CommandlineFeedback


def _run_decision_experiment() -> None:
    arms = [
        "The best arm.",  # 0
        "The second best arm.",  # 1
        "The other second best arm, flip a coin.",  # 2
        "Third best arm.",  # 3
        "Least favorite arm.",  # 4
    ]
    feedback_mechanism = CommandlineFeedback(len(arms))
    savage(feedback_mechanism=feedback_mechanism, delta=0.5, verbose=True)


if __name__ == "__main__":
    _run_decision_experiment()
