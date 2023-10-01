import math
import torch
from linguamechanica.se3 import ImplicitDualQuaternion


class TestImplicitDualQuaternion:
    coords = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, torch.pi / 2.0, 0, 0],
            [0, 0, 0, torch.pi / 4.0, 0, 0],
            [0, 0, 0, 0, torch.pi / 4.0, 0],
            [0, 0, 0, 0, 0, torch.pi / 4.0],
            [1, 0, 0, torch.pi / 4.0, 0, 0],
        ]
    ).float()
    expected_exp = torch.tensor(
        [
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 1, 0, 2, 0],
            [0, 0, 0, 1, 0, 0, 2],
            [math.sin(math.pi / 2.0), 0, 0, math.cos(math.pi / 2.0), 0, 0, 0],
            [math.sin(math.pi / 4.0), 0, 0, math.cos(math.pi / 4.0), 0, 0, 0],
            [0, math.sin(math.pi / 4.0), 0, math.cos(math.pi / 4.0), 0, 0, 0],
            [0, 0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0), 0, 0, 0],
            [math.sin(math.pi / 4.0), 0, 0, math.cos(math.pi / 4.0), 2, 0, 0],
        ],
        requires_grad=True,
    ).float()

    expected_exp_squared = torch.tensor(
        [
            [0, 0, 0, 1, 4, 0, 0],
            [0, 0, 0, 1, 0, 4, 0],
            [0, 0, 0, 1, 0, 0, 4],
            [math.sin(math.pi), 0, 0, math.cos(math.pi), 0, 0, 0],
            [math.sin(math.pi / 2.0), 0, 0, math.cos(math.pi / 2.0), 0, 0, 0],
            [0, math.sin(math.pi / 2.0), 0, math.cos(math.pi / 2.0), 0, 0, 0],
            [0, 0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0, 0, 0],
            [math.sin(math.pi / 2.0), 0, 0, math.cos(math.pi / 2.0), 4, 0, 0],
        ]
    ).float()

    epsilon = 1e-6

    def test_exp(self):
        se3 = ImplicitDualQuaternion()
        se3_idq = se3.exp(self.coords)
        assert (self.expected_exp - se3_idq).abs().mean(1).mean(0).item() < self.epsilon

    def test_log(self):
        se3 = ImplicitDualQuaternion()
        se3_idq = se3.exp(self.coords)
        se3_log = se3.log(se3_idq)
        assert (self.coords - se3_log).abs().mean(1).mean(0).item() < self.epsilon

    def test_chain(self):
        se3 = ImplicitDualQuaternion()
        se3_idq = se3.exp(self.coords)
        exp_squared = se3.chain(se3_idq, se3_idq)
        assert (self.expected_exp_squared - exp_squared).abs().mean(1).mean(
            0
        ).item() < self.epsilon

    def test_invert(self):
        se3 = ImplicitDualQuaternion()
        exp_squared_inv = se3.invert(self.expected_exp_squared)
        zeros = se3.log(se3.chain(exp_squared_inv, self.expected_exp_squared))
        assert (zeros - torch.zeros_like(zeros)).abs().mean(1).mean(
            0
        ).item() < self.epsilon

    def test_identity(self):
        se3 = ImplicitDualQuaternion()
        identities = se3.identity(self.expected_exp_squared.shape[0])
        exp_squared = se3.chain(
            se3.chain(identities, self.expected_exp_squared), identities
        )
        assert (exp_squared - self.expected_exp_squared).abs().mean(1).mean(
            0
        ).item() < self.epsilon
