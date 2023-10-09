import math
import torch
from linguamechanica.se3 import ImplicitDualQuaternion, ProjectiveMatrix
import pytest

se3_representations = [
    ProjectiveMatrix(),
    ImplicitDualQuaternion(),
]


class TestSE3:
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

    coords_square = torch.tensor(
        [
            [2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, torch.pi, 0, 0],
            [0, 0, 0, torch.pi / 2.0, 0, 0],
            [0, 0, 0, 0, torch.pi / 2.0, 0],
            [0, 0, 0, 0, 0, torch.pi / 2.0],
            [2, 0, 0, torch.pi / 2.0, 0, 0],
        ]
    ).float()

    expected_exp_idq = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [math.sin(math.pi / 4.0), 0, 0, math.cos(math.pi / 4.0), 0, 0, 0],
            [math.sin(math.pi / 8.0), 0, 0, math.cos(math.pi / 8.0), 0, 0, 0],
            [0, math.sin(math.pi / 8.0), 0, math.cos(math.pi / 8.0), 0, 0, 0],
            [0, 0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0), 0, 0, 0],
            [math.sin(math.pi / 8.0), 0, 0, math.cos(math.pi / 8.0), 1, 0, 0],
        ],
        requires_grad=True,
    ).float()

    epsilon = 1e-6

    def test_exp_idq(self):
        se3 = ImplicitDualQuaternion()
        element = se3.exp(self.coords)
        computed_zero_pose = se3.log(
            se3.chain(se3.invert(self.expected_exp_idq), element)
        )
        assert computed_zero_pose.abs().mean().item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_exp_log_1(self, se3):
        ImplicitDualQuaternion()
        computed_coord = se3.log(se3.exp(self.coords))
        assert (self.coords[:, :3] - computed_coord[:, :3]).abs().mean(1).mean(
            0
        ).item() < self.epsilon
        assert (self.coords[:, 3:].cos() - computed_coord[:, 3:].cos()).abs().mean(
            1
        ).mean(0).item() < self.epsilon
        assert (self.coords[:, 3:].sin() - computed_coord[:, 3:].sin()).abs().mean(
            1
        ).mean(0).item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_exp_log_2(self, se3):
        se3_idq = se3.exp(self.coords)
        se3_log = se3.log(se3_idq)
        assert (self.coords - se3_log).abs().mean(1).mean(0).item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_log_rot_trans(self, se3):
        pose_rot1 = torch.tensor(
            [
                [0, 0, 0, 0, 0, torch.pi / 2.0],
                [0, 0, 0, 0, 0, torch.pi / 2.0],
                [0, 0, 0, 0, 0, torch.pi / 2.0],
            ]
        ).float()
        pose_trans1 = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ]
        ).float()
        se3_batch = se3.chain(se3.exp(pose_rot1), se3.exp(pose_trans1))
        assert (se3.exp(se3.log(se3_batch)) - se3_batch).abs().sum() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_chain(self, se3):
        se3_idq = se3.exp(self.coords)
        squared = se3.log(se3.chain(se3_idq, se3_idq))
        assert (self.coords_square[:, :3] - squared[:, :3]).abs().mean(1).mean(
            0
        ).item() < self.epsilon
        assert (self.coords_square[:, 3:].cos() - squared[:, 3:].cos()).abs().mean(
            1
        ).mean(0).item() < self.epsilon
        assert (self.coords_square[:, 3:].sin() - squared[:, 3:].sin()).abs().mean(
            1
        ).mean(0).item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_invert(self, se3):
        exp_squared_inv = se3.invert(se3.exp(self.coords))
        zeros = se3.log(se3.chain(exp_squared_inv, se3.exp(self.coords)))
        assert (zeros - torch.zeros_like(zeros)).abs().mean(1).mean(
            0
        ).item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_identity(self, se3):
        identities = se3.identity(self.coords.shape[0])
        chained = se3.chain(se3.chain(identities, se3.exp(self.coords)), identities)
        assert (chained - se3.exp(self.coords)).abs().mean().item() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_act_vector(self, se3):
        vectors = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).float()
        poses = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, torch.pi]]).float()
        expected_acted_vectors = torch.tensor(
            [
                # Vectors have no effect on translation
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            ]
        ).float()
        acted_vector = se3.act_vector(se3.exp(poses), vectors)
        assert (expected_acted_vectors - acted_vector).abs().mean() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_act_vector_two_rot(self, se3):
        vectors = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).float()
        pose_rot1 = torch.tensor([[0, 0, 0, 0, 0, torch.pi / 2.0]]).float()
        pose_rot2 = torch.tensor(
            [
                [0, 0, 0, torch.pi / 2.0, 0, 0],
            ]
        ).float()

        expected_acted_vectors = torch.tensor(
            [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]
        ).float()
        se3_rot_trans = se3.chain(se3.exp(pose_rot1), se3.exp(pose_rot2))
        acted_vector = se3.act_vector(se3_rot_trans, vectors)
        assert (expected_acted_vectors - acted_vector).abs().mean() < self.epsilon

    @pytest.mark.parametrize("se3", se3_representations)
    def test_act_points_rot_trans(self, se3):
        points = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).float()
        pose_rot1 = torch.tensor([[0, 0, 0, 0, 0, torch.pi / 2.0]]).float()
        pose_trans1 = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
            ]
        ).float()
        expected_acted_points = torch.tensor(
            [[[0, 2, 0], [-1, 1, 0], [0, 1, 1]]]
        ).float()
        se3_batch = se3.chain(se3.exp(pose_rot1), se3.exp(pose_trans1))
        acted_points = se3.act_point(se3_batch, points)
        assert (expected_acted_points - acted_points).abs().mean() < self.epsilon
