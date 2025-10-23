import logging

import numpy as np

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


class EarlyStoppingMinimize:
    """Simple early stopping implementation based on relative improvement of the validation loss.

    Training stops if no significant relative improvement is observed over a given patience period.
    Handles only metrics that should be minimized. Simply multiply metrics to be maximized by -1 at
    runtime.
    """

    def __init__(
        self,
        patience: int = 10,
        rel_delta: float = 0.01,
    ):
        """
        Args:
            patience (int): Number of epochs to wait for a significant improvement.
                            Default: 3
            rel_delta (float): Minimum relative improvement (as a fraction) to qualify as progress.
                               Default: 0.01 (1%)
        """
        if patience < 1:
            raise ValueError("Patience must be at least 1")
        if rel_delta < 0:
            raise ValueError("Relative delta must be non-negative")

        self.patience = patience
        self.rel_delta = rel_delta
        self.reset()

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should be stopped based on the current metric value.

        Args:
            current_value (float): Current metric value (loss or other metric).

        Returns:
            bool: True if no significant improvement is seen for 'patience' consecutive calls,
                  indicating that training should be stopped; False otherwise.
        """
        # Handle NaN values
        if not isinstance(current_value, (int, float)) or (
            isinstance(current_value, float) and not np.isfinite(current_value)
        ):
            logger.warning(
                f"Received invalid value: {current_value}. Counting as no improvement."
            )
            self.counter += 1
            return self.counter >= self.patience

        # First evaluation or reset
        if self.best_score is None:
            self.best_score = current_value
            return False

        # Calculate improvement
        if abs(self.best_score) > 1e-10:  # Avoid division by zero
            improvement = (self.best_score - current_value) / abs(self.best_score)
        else:
            improvement = self.best_score - current_value

        if improvement >= self.rel_delta:
            logger.debug(
                f"Metric decreased by "
                f"{abs(improvement) * 100:.2f}%: {self.best_score:.6f} -> {current_value:.6f}"
            )
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"No significant improvement: {'increase' if improvement < 0 else 'decrease'} of "
                f"{abs(improvement) * 100:.2f}% ({self.best_score:.6f} -> {current_value:.6f}). "
                f"Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                self.early_stop = True
                return True

        return False
