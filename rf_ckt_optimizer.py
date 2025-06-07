import torch
import numpy as np
import skrf as rf
from typing import List, Union


class RfOptNtwk():
    def __init__(self, frequency: torch.Tensor, z0: float = 50.):
        self.frequency = frequency
        self.z0 = z0
        self.w: torch.Tensor = self.frequency * 2 * np.pi
        self.s_mat: torch.Tensor = torch.zeros(
            (self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        self.opt_vars: List[torch.Tensor] = []
        self.opt_lr: List[float] = []


class Splitter(RfOptNtwk):
    def __init__(self, frequency: torch.Tensor, z0: float = 50., num_port: int = 3):
        super(Splitter, self).__init__(frequency, z0)
        self.s_mat = ((2.0/num_port-1)*torch.eye(num_port, dtype=torch.complex64)+(2/num_port)
                      * (torch.ones(num_port)-torch.eye(num_port))).repeat(frequency.shape[0], 1, 1)
        
class Ground(RfOptNtwk):
    def __init__(self, frequency: torch.Tensor, z0:float =50):
        super(Ground, self).__init__(frequency, z0)
        self.s_mat = torch.zeros_like(frequency)


class RfOptMath():
    @classmethod
    def cascade(cls, ntwk1: RfOptNtwk, ntwk2: RfOptNtwk):
        if ntwk1.frequency.shape != ntwk2.frequency.shape:
            raise ValueError(
                "Frequency vectors must match for cascading networks")
        if ntwk1.s_mat.shape[1] != 2 or ntwk2.s_mat.shape[1] != 2:
            raise ValueError("cascade can use only 2port network")
        new_ntwk = RfOptMath.connect_s(ntwk1,1,ntwk2,2)
        return new_ntwk

    @classmethod
    def parallel(cls, ntwk1: RfOptNtwk, ntwk2: RfOptNtwk):
        if ntwk1.frequency.shape != ntwk2.frequency.shape:
            raise ValueError(
                "Frequency vectors must match for parallel networks")

        splitter = Splitter(ntwk1.frequency, ntwk1.z0, 3)

        result = RfOptMath.connect_s(splitter, 1, ntwk1, 0)
        result = RfOptMath.connect_s(result, 1, ntwk2, 0)
        
        result = RfOptMath.connect_s(result,1,splitter,0)
        result = RfOptMath.innerconnect_s(result,1,2)

        return result

    @classmethod
    def connect_s(cls, ntwk1: RfOptNtwk, l: int, ntwk2: RfOptNtwk, k: int) -> RfOptNtwk:
        if ntwk1.frequency.shape != ntwk2.frequency.shape:
            raise ValueError(
                "Frequency vectors must match for connecting networks")
        if l > ntwk1.s_mat.shape[-1]-1 or k > ntwk2.s_mat.shape[-1]-1:
            raise ValueError("port indices are out of range")

        s_mat1 = ntwk1.s_mat
        s_mat2 = ntwk2.s_mat

        nf = s_mat1.shape[0]  # num frequency points
        nA = s_mat1.shape[1]  # num ports on A
        nB = s_mat2.shape[1]  # num ports on B
        nC = nA + nB  # num ports on C

        new_s_mat = torch.zeros((nf, nC, nC), dtype=torch.complex64)
        new_s_mat[:, :nA, :nA] = s_mat1
        new_s_mat[:, nA:, nA:] = s_mat2

        new_ntwk = RfOptNtwk(ntwk1.frequency, ntwk1.z0)
        new_ntwk.s_mat = new_s_mat

        return cls.innerconnect_s(new_ntwk, k, nA+l)

    @classmethod
    def innerconnect_s(cls, ntwk: RfOptNtwk, k: int, l: int) -> RfOptNtwk:
        s_mat = ntwk.s_mat
        if k > s_mat.shape[-1]-1 or l > s_mat.shape[-1]-1:
            raise (ValueError("port indices are out of range"))
        if k == l:
            raise (ValueError("port indices must be different"))

        nA = s_mat.shape[1]  # num of ports on input s-matrix
        ext_i = [i for i in range(nA) if i not in (k, l)]

        s_mat_kl = 1.0 - s_mat[:, k, l]
        s_mat_lk = 1.0 - s_mat[:, l, k]
        s_mat_kk = s_mat[:, k, k]
        s_mat_ll = s_mat[:, l, l]

        det = (s_mat_kk * s_mat_ll - s_mat_kl * s_mat_lk)

        if torch.allclose(det, torch.zeros_like(det)):
            raise ValueError(
                "Determinant is zero, cannot perform inner connection")

        s_mat_ke = s_mat[:, k, ext_i].T
        s_mat_le = s_mat[:, l, ext_i].T
        s_mat_ek = s_mat[:, ext_i, k].T
        s_mat_el = s_mat[:, ext_i, l].T

        i, j = torch.meshgrid(ext_i, ext_i, indexing='ij')
        i = i.to_sparse()
        j = j.to_sparse()
        new_s_mat = s_mat[:, i, j]

        tmp_a = s_mat_el * (s_mat_lk / det) + s_mat_ek * (s_mat_ll / det)
        tmp_b = s_mat_el * (s_mat_kk / det) + s_mat_ek * (s_mat_kl / det)

        for i in range(nA-2):
            new_s_mat[:, i, :] += (s_mat_ke*tmp_a[i] + s_mat_le*tmp_b[i]).T
        new_ntwk = RfOptNtwk(ntwk.frequency, ntwk.z0)
        new_ntwk.s_mat = new_s_mat

        return new_ntwk


class RFCircuitOptimizer():
    def __init__(self, frequency: torch.Tensor,
                 z0_port: float = 50.):
        self.frequency = frequency
        self.z0_port = z0_port
        self.w = self.frequency * 2 * np.pi

    def capacitor(self, cap: torch.Tensor):
        s_mat = torch.zeros(
            (self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        z0_0 = 50.0+0.0j
        z0_1 = 50.0+0.0j
        denom = 1.0 + 1j * self.w * cap * (z0_0 + z0_1)
        s_mat[:, 0, 0] = (1.0 - 1j * self.w * cap *
                          (z0_0.conjugate() - z0_1)) / denom
        s_mat[:, 1, 1] = (1.0 - 1j * self.w * cap *
                          (z0_1.conjugate() - z0_0)) / denom
        s_mat[:, 0, 1] = (2j * self.w * cap *
                          (z0_0.real * z0_1.real)**0.5) / denom
        s_mat[:, 1, 0] = (2j * self.w * cap *
                          (z0_0.real * z0_1.real)**0.5) / denom
        return s_mat

    def inductor(self, ind: torch.Tensor):
        s_mat = torch.zeros(
            (self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        z0_0 = 50.0+0.0j
        z0_1 = 50.0+0.0j
        denom = (1j * self.w * ind) + (z0_0 + z0_1)
        s_mat[:, 0, 0] = (1j * self.w * ind - z0_0.conjugate() + z0_1) / denom
        s_mat[:, 1, 1] = (1j * self.w * ind + z0_0 - z0_1.conjugate()) / denom
        s_mat[:, 0, 1] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        s_mat[:, 1, 0] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        return s_mat

    def resistor(self, res: torch.Tensor):
        s_mat = torch.zeros(
            (self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        z0_0 = 50.0+0.0j
        z0_1 = 50.0+0.0j
        denom = (z0_0 + z0_1) + res
        s_mat[:, 0, 0] = (res - z0_0.conjugate() + z0_1) / denom
        s_mat[:, 1, 1] = (res + z0_0 - z0_1.conjugate()) / denom
        s_mat[:, 0, 1] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        s_mat[:, 1, 0] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        return s_mat


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, s_mat: torch.Tensor, target_s_mat: torch.Tensor):
        if s_mat.shape != target_s_mat.shape:
            raise ValueError("Shape mismatch between s_mat and target_s_mat")
        return torch.mean(torch.abs(s_mat - target_s_mat))
