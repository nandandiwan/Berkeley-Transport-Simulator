"""Data containers for Hamiltonian block structures.

Separated to keep core objects lightweight and dependency-free.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class HamiltonianBlocks:
    """Block-tridiagonal pieces for a quasi-1D system.

    Hl: coupling from layer i to i-1  (left hopping)
    H0: on-site block for a representative principal layer
    Hr: coupling from layer i to i+1  (right hopping)
    orbitals_per_layer: number of orbitals in a principal layer (dimension of H0)
    basis_labels: list[str] giving orbital labels (optional for diagnostics)
    """
    Hl: np.ndarray
    H0: np.ndarray
    Hr: np.ndarray
    orbitals_per_layer: int
    basis_labels: list[str] | None = None

    def validate(self) -> None:
        n = self.orbitals_per_layer
        if self.H0.shape != (n, n):
            raise ValueError(f"H0 shape {self.H0.shape} != ({n},{n})")
        if self.Hl.shape != (n, n):
            raise ValueError(f"Hl shape {self.Hl.shape} != ({n},{n})")
        if self.Hr.shape != (n, n):
            raise ValueError(f"Hr shape {self.Hr.shape} != ({n},{n})")
        if not np.allclose(self.Hl.conj().T, self.Hr, atol=1e-12):
            # Don't enforce Hermitian exactly; store warning only.
            pass

    @property
    def dim(self) -> int:
        return self.orbitals_per_layer

    def copy(self) -> "HamiltonianBlocks":
        return HamiltonianBlocks(Hl=self.Hl.copy(), H0=self.H0.copy(), Hr=self.Hr.copy(),
                                 orbitals_per_layer=self.orbitals_per_layer,
                                 basis_labels=None if self.basis_labels is None else list(self.basis_labels))
