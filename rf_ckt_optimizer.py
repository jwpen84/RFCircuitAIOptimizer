import torch
import numpy as np

class RFCircuitOptimizer():
    def __init__(self, frequency: torch.Tensor,
                 z0_port: float = 50.):
        self.frequency = frequency
        self.z0_port = z0_port
        self.w = self.frequency * 2 * np.pi

    def capacitor(self, cap: torch.Tensor):
        s_mat = torch.zeros((self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        z0_0 = 50.0+0.0j
        z0_1 = 50.0+0.0j
        denom = 1.0 + 1j * self.w * cap * (z0_0 + z0_1)
        s_mat[:, 0, 0] = (1.0 - 1j * self.w * cap * (z0_0.conjugate() - z0_1) ) / denom
        s_mat[:, 1, 1] = (1.0 - 1j * self.w * cap * (z0_1.conjugate() - z0_0) ) / denom
        s_mat[:, 0, 1] = (2j * self.w * cap * (z0_0.real * z0_1.real)**0.5) / denom
        s_mat[:, 1, 0] = (2j * self.w * cap * (z0_0.real * z0_1.real)**0.5) / denom
        return s_mat
    
    def inductor(self, ind: torch.Tensor):
        s_mat = torch.zeros((self.frequency.shape[0], 2, 2), dtype=torch.complex64)
        z0_0 = 50.0+0.0j
        z0_1 = 50.0+0.0j
        denom = (1j * self.w * ind) + (z0_0 + z0_1)
        s_mat[:, 0, 0] = (1j * self.w * ind - z0_0.conjugate() + z0_1) / denom
        s_mat[:, 1, 1] = (1j * self.w * ind + z0_0 - z0_1.conjugate()) / denom
        s_mat[:, 0, 1] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        s_mat[:, 1, 0] = 2 * (z0_0.real * z0_1.real)**0.5 / denom
        return s_mat
    
    def resistor(self, res: torch.Tensor):
        s_mat = torch.zeros((self.frequency.shape[0], 2, 2), dtype=torch.complex64)
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
    
    # def grad(self, s_mat: torch.Tensor, target_s_mat: torch.Tensor):
    #     if s_mat.shape != target_s_mat.shape:
    #         raise ValueError("Shape mismatch between s_mat and target_s_mat")
    #     return 2 * (s_mat - target_s_mat) / s_mat.numel()