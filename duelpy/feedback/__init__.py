"""Various mechanisms for comparing arms."""

from duelpy.feedback.adversarial_matrix_feedback import AdversarialMatrixFeedback
from duelpy.feedback.arm_feature_feedback import ArmFeatureFeedback
from duelpy.feedback.commandline_feedback import CommandlineFeedback
from duelpy.feedback.contextual_feedback_mechanism import ContextualFeedbackMechanism
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.feedback.linear_context_feedback import LinearContextFeedback
from duelpy.feedback.matrix_feedback import MatrixFeedback

__all__ = [
    "AdversarialMatrixFeedback",
    "ArmFeatureFeedback",
    "CommandlineFeedback",
    "ContextualFeedbackMechanism",
    "FeedbackMechanism",
    "LinearContextFeedback",
    "MatrixFeedback",
]
