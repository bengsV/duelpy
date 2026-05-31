"""Various mechanisms for comparing arms."""

from duelpy.feedback.adversarial_matrix_feedback import AdversarialMatrixFeedback
from duelpy.feedback.commandline_feedback import CommandlineFeedback
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.feedback.matrix_feedback import MatrixFeedback

__all__ = [
    "AdversarialMatrixFeedback",
    "CommandlineFeedback",
    "FeedbackMechanism",
    "MatrixFeedback",
]
