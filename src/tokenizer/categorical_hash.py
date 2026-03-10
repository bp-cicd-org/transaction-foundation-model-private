"""
Hash-based categorical tokenizer (GPU-accelerated).

Maps high-cardinality string columns (e.g. merchant names) to a fixed number
of hash buckets via modulo.  Expects the caller to pre-hash the column to
integers before calling tokenize() (e.g. cudf.Series.hash_values()).

Data is never stored in the constructor.
"""

import cudf
import cupy as cp

from .base import BaseTokenizer


class CategoricalHashTokenizer(BaseTokenizer):
    """Hash-modulo tokenizer for high-cardinality categoricals."""

    def __init__(
        self,
        vocab_limit: int = 2048,
        special_token: str = "MERCH",
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.vocab_limit = vocab_limit
        self.special_token = special_token
        self.stream = stream
        self._vocab_built = False

    def build_vocab(self, column_data=None) -> None:
        """Create ``{0: "MERCH_0", 1: "MERCH_1", ...}``."""
        if self.stream:
            with self.stream:
                idx = cudf.Series(range(self.vocab_limit))
                tokens = cudf.Series(
                    [self.special_token + "_"] * self.vocab_limit
                ).str.cat(idx.astype(str), sep="")
                self._idx_to_token = dict(
                    zip(idx.to_pandas(), tokens.to_pandas())
                )
        else:
            self._idx_to_token = {
                i: f"{self.special_token}_{i}" for i in range(self.vocab_limit)
            }
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        """*column_data* must already be integer hash values."""
        token_bucket = column_data % self.vocab_limit
        return token_bucket.map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"CategoricalHashTokenizer(token={self.special_token}, "
            f"limit={self.vocab_limit}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "special_token": self.special_token,
            "vocab_limit": self.vocab_limit,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {"_vocab_built": self._vocab_built}

    def _set_fitted_state(self, state: dict) -> None:
        self._vocab_built = state.get("_vocab_built", False)
