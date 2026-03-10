"""
Log-compressed time-delta tokenizer (GPU-accelerated).

Bins time differences (in seconds) into logarithmically-spaced buckets.
Useful for capturing recency signals — small deltas (seconds/minutes) get
fine-grained bins while large deltas (months) share coarser bins.

Data is never stored in the constructor.
"""

import cudf
import cupy as cp

from .base import BaseTokenizer

_SECONDS_PER_JULIAN_YEAR = 31556951.999999996


class TimeDeltaTokenizer(BaseTokenizer):
    """Log-scale binning for time deltas (in seconds)."""

    def __init__(
        self,
        num_bins: int = 32,
        special_token: str = "TDIF",
        max_years: float = 10.0,
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.special_token = special_token
        self.max_years = max_years
        self.stream = stream

        self.max_horizon = int(max_years * _SECONDS_PER_JULIAN_YEAR)
        log_max = cp.log(float(self.max_horizon) + 1.0)
        self.boundaries = cp.linspace(0, log_max, self.num_bins + 1)
        self._vocab_built = False

    def build_vocab(self, column_data=None) -> None:
        if self.stream:
            with self.stream:
                self._idx_to_token = {
                    i: f"{self.special_token}_{i}" for i in range(self.num_bins)
                }
        else:
            self._idx_to_token = {
                i: f"{self.special_token}_{i}" for i in range(self.num_bins)
            }
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        if self.stream:
            with self.stream:
                return self._tokenize_internal(column_data)
        return self._tokenize_internal(column_data)

    def _tokenize_internal(self, column_data) -> cudf.Series:
        clamped = column_data.clip(0, self.max_horizon)
        clamped_f64 = clamped.values.astype(cp.float64)
        log_vals = cp.log(clamped_f64 + 1.0)
        token_ids = cp.clip(
            cp.digitize(log_vals, self.boundaries), 0, self.num_bins - 1
        )
        return cudf.Series(token_ids, index=column_data.index).map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"TimeDeltaTokenizer(token={self.special_token}, "
            f"bins={self.num_bins}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "special_token": self.special_token,
            "max_years": self.max_years,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "boundaries": (
                self.boundaries.get()
                if isinstance(self.boundaries, cp.ndarray)
                else self.boundaries
            ),
            "max_horizon": self.max_horizon,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        self.boundaries = cp.array(state["boundaries"])
        self.max_horizon = state["max_horizon"]
        self._vocab_built = state.get("_vocab_built", False)
