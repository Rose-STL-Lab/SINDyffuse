from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
ArrayLike = np.ndarray

@dataclass(frozen=True)
class ThetaSpec:
    tier: str = 'tier1_small'
    max_cross_terms: Optional[int] = None
    include_bias: bool = True
    include_linear_u: bool = True
    include_linear_c: bool = True
    include_u_times_c: bool = False
    include_contact_gated: bool = False
    include_phase_terms: bool = False

class ThetaLibrary:

    def __init__(self, spec: ThetaSpec=ThetaSpec()):
        self.spec = spec

    @staticmethod
    def _as_2d(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr.reshape(-1, arr.shape[-1])
        raise ValueError(f'Expected array with ndim 1,2,3 got {arr.shape}')

    @staticmethod
    def _pairwise_terms(x: np.ndarray, names: Sequence[str], max_terms: Optional[int]=None) -> Tuple[np.ndarray, List[str]]:
        pieces: List[np.ndarray] = []
        out_names: List[str] = []
        count = 0
        n = int(x.shape[1])
        for i in range(n):
            for j in range(i, n):
                pieces.append((x[:, i] * x[:, j])[:, None])
                out_names.append(f'{names[i]}*{names[j]}')
                count += 1
                if max_terms is not None and count >= int(max_terms):
                    return (np.hstack(pieces).astype(np.float32), out_names)
        if not pieces:
            return (np.zeros((x.shape[0], 0), dtype=np.float32), [])
        return (np.hstack(pieces).astype(np.float32), out_names)

    @staticmethod
    def _cross_terms(a: np.ndarray, a_names: Sequence[str], b: np.ndarray, b_names: Sequence[str], max_terms: Optional[int]=None) -> Tuple[np.ndarray, List[str]]:
        pieces: List[np.ndarray] = []
        out_names: List[str] = []
        count = 0
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                pieces.append((a[:, i] * b[:, j])[:, None])
                out_names.append(f'{a_names[i]}*{b_names[j]}')
                count += 1
                if max_terms is not None and count >= int(max_terms):
                    return (np.hstack(pieces).astype(np.float32), out_names)
        if not pieces:
            return (np.zeros((a.shape[0], 0), dtype=np.float32), [])
        return (np.hstack(pieces).astype(np.float32), out_names)

    @staticmethod
    def _phase_terms(u: np.ndarray, u_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        idxs = [i for i, n in enumerate(u_names) if n in {'phase_sin', 'phase_cos'}]
        if not idxs:
            return (np.zeros((u.shape[0], 0), dtype=np.float32), [])
        phase = u[:, idxs]
        return (phase.astype(np.float32), [u_names[i] for i in idxs])

    def build(self, u: Optional[ArrayLike]=None, c: Optional[ArrayLike]=None, u_names: Optional[Sequence[str]]=None, c_names: Optional[Sequence[str]]=None) -> Tuple[np.ndarray, List[str]]:
        u2 = self._as_2d(u)
        c2 = self._as_2d(c)
        reference = u2 if u2 is not None else c2
        if reference is None:
            raise ValueError('At least one of u or c is required')
        if u2 is not None and u_names is None:
            u_names = [f'u{i}' for i in range(u2.shape[1])]
        if c2 is not None and c_names is None:
            c_names = [f'c{i}' for i in range(c2.shape[1])]
        pieces: List[np.ndarray] = []
        names: List[str] = []
        if self.spec.include_bias:
            pieces.append(np.ones((reference.shape[0], 1), dtype=np.float32))
            names.append('1')
        if u2 is not None and self.spec.include_linear_u:
            pieces.append(u2.astype(np.float32))
            names.extend([f'u:{n}' for n in u_names])
        if c2 is not None and self.spec.include_linear_c:
            pieces.append(c2.astype(np.float32))
            names.extend([f'c:{n}' for n in c_names])
        tier = str(self.spec.tier).strip().lower()
        max_cross_terms = self.spec.max_cross_terms
        if u2 is not None and c2 is not None and self.spec.include_u_times_c:
            cross, cn = self._cross_terms(u2, u_names, c2, c_names, max_terms=max_cross_terms)
            if cross.shape[1] > 0:
                pieces.append(cross)
                names.extend([f'u*c:{n}' for n in cn])
        if self.spec.include_phase_terms and u2 is not None and (u_names is not None):
            phase, pn = self._phase_terms(u2, u_names)
            if phase.shape[1] > 0:
                pieces.append(phase)
                names.extend([f'phase:{n}' for n in pn])
        if tier == 'tier3_contact_periodic' and c2 is not None:
            state_parts: List[np.ndarray] = []
            state_names: List[str] = []
            if u2 is not None:
                state_parts.append(u2)
                state_names.extend([f'u:{n}' for n in u_names])
            if c2 is not None:
                state_parts.append(c2)
                state_names.extend([f'c:{n}' for n in c_names])
            state = np.hstack(state_parts).astype(np.float32) if state_parts else None
            gated_parts: List[np.ndarray] = []
            gated_names: List[str] = []
            gate_indices = [0, 1, 2, 3]
            if c2.shape[1] > 13:
                gate_indices.append(13)
            if c2.shape[1] > 14:
                gate_indices.append(14)
            if state is not None:
                for gate_idx in gate_indices:
                    if gate_idx < c2.shape[1]:
                        gate = c2[:, gate_idx:gate_idx + 1]
                        gate_name = c_names[gate_idx] if c_names and gate_idx < len(c_names) else f'c{gate_idx}'
                        gated_parts.append(gate * state)
                        gated_names.extend([f'cgate[{gate_name}]*{name}' for name in state_names])
            if gated_parts:
                pieces.append(np.hstack(gated_parts).astype(np.float32))
                names.extend(gated_names)
        if not pieces:
            return (np.zeros((reference.shape[0], 0), dtype=np.float32), [])
        theta = np.hstack(pieces).astype(np.float32)
        return (theta, names)

    @staticmethod
    def _as_2d_torch(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.dim() == 1:
            return x.unsqueeze(0)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        raise ValueError(f'Expected tensor with ndim 1,2,3 got {x.shape}')

    @staticmethod
    def _pairwise_terms_torch(x: torch.Tensor, names: Sequence[str], max_terms: Optional[int]=None) -> Tuple[torch.Tensor, List[str]]:
        pieces: List[torch.Tensor] = []
        out_names: List[str] = []
        count = 0
        n = int(x.shape[1])
        for i in range(n):
            for j in range(i, n):
                pieces.append((x[:, i] * x[:, j]).unsqueeze(1))
                out_names.append(f'{names[i]}*{names[j]}')
                count += 1
                if max_terms is not None and count >= int(max_terms):
                    return (torch.cat(pieces, dim=1), out_names)
        if not pieces:
            return (x.new_zeros((x.shape[0], 0)), [])
        return (torch.cat(pieces, dim=1), out_names)

    @staticmethod
    def _cross_terms_torch(a: torch.Tensor, a_names: Sequence[str], b: torch.Tensor, b_names: Sequence[str], max_terms: Optional[int]=None) -> Tuple[torch.Tensor, List[str]]:
        pieces: List[torch.Tensor] = []
        out_names: List[str] = []
        count = 0
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                pieces.append((a[:, i] * b[:, j]).unsqueeze(1))
                out_names.append(f'{a_names[i]}*{b_names[j]}')
                count += 1
                if max_terms is not None and count >= int(max_terms):
                    return (torch.cat(pieces, dim=1), out_names)
        if not pieces:
            return (a.new_zeros((a.shape[0], 0)), [])
        return (torch.cat(pieces, dim=1), out_names)

    @staticmethod
    def _phase_terms_torch(u: torch.Tensor, u_names: Sequence[str]) -> Tuple[torch.Tensor, List[str]]:
        idxs = [i for i, n in enumerate(u_names) if n in {'phase_sin', 'phase_cos'}]
        if not idxs:
            return (u.new_zeros((u.shape[0], 0)), [])
        phase = u[:, idxs]
        return (phase, [u_names[i] for i in idxs])

    def build_torch(self, u: Optional[torch.Tensor]=None, c: Optional[torch.Tensor]=None, u_names: Optional[Sequence[str]]=None, c_names: Optional[Sequence[str]]=None) -> Tuple[torch.Tensor, List[str]]:
        u2 = self._as_2d_torch(u)
        c2 = self._as_2d_torch(c)
        reference = u2 if u2 is not None else c2
        if reference is None:
            raise ValueError('At least one of u or c is required')
        if u2 is not None and u_names is None:
            u_names = [f'u{i}' for i in range(u2.shape[1])]
        if c2 is not None and c_names is None:
            c_names = [f'c{i}' for i in range(c2.shape[1])]
        pieces: List[torch.Tensor] = []
        names: List[str] = []
        if self.spec.include_bias:
            pieces.append(torch.ones((reference.shape[0], 1), dtype=reference.dtype, device=reference.device))
            names.append('1')
        if u2 is not None and self.spec.include_linear_u:
            pieces.append(u2)
            names.extend([f'u:{n}' for n in u_names])
        if c2 is not None and self.spec.include_linear_c:
            pieces.append(c2)
            names.extend([f'c:{n}' for n in c_names])
        tier = str(self.spec.tier).strip().lower()
        max_cross_terms = self.spec.max_cross_terms
        if u2 is not None and c2 is not None and self.spec.include_u_times_c:
            cross, cn = self._cross_terms_torch(u2, u_names, c2, c_names, max_terms=max_cross_terms)
            if cross.shape[1] > 0:
                pieces.append(cross)
                names.extend([f'u*c:{n}' for n in cn])
        if self.spec.include_phase_terms and u2 is not None and (u_names is not None):
            phase, pn = self._phase_terms_torch(u2, u_names)
            if phase.shape[1] > 0:
                pieces.append(phase)
                names.extend([f'phase:{n}' for n in pn])
        if tier == 'tier3_contact_periodic' and c2 is not None:
            state_parts: List[torch.Tensor] = []
            state_names: List[str] = []
            if u2 is not None:
                state_parts.append(u2)
                state_names.extend([f'u:{n}' for n in u_names])
            if c2 is not None:
                state_parts.append(c2)
                state_names.extend([f'c:{n}' for n in c_names])
            state = torch.cat(state_parts, dim=1) if state_parts else None
            gated_parts: List[torch.Tensor] = []
            gated_names: List[str] = []
            gate_indices = [0, 1, 2, 3]
            if c2.shape[1] > 13:
                gate_indices.append(13)
            if c2.shape[1] > 14:
                gate_indices.append(14)
            if state is not None:
                for gate_idx in gate_indices:
                    if gate_idx < c2.shape[1]:
                        gate = c2[:, gate_idx:gate_idx + 1]
                        gate_name = c_names[gate_idx] if c_names and gate_idx < len(c_names) else f'c{gate_idx}'
                        gated_parts.append(gate * state)
                        gated_names.extend([f'cgate[{gate_name}]*{name}' for name in state_names])
            if gated_parts:
                pieces.append(torch.cat(gated_parts, dim=1))
                names.extend(gated_names)
        if not pieces:
            return (torch.zeros((reference.shape[0], 0), dtype=reference.dtype, device=reference.device), [])
        theta = torch.cat(pieces, dim=1)
        return (theta, names)