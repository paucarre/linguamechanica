import math
import numpy as np
from linguamechanica.kinematics import (
    DifferentiableOpenChainMechanism,
    UrdfRobotLibrary,
    to_left_multiplied,
)
import random
import torch
from linguamechanica.se3 import ProjectiveMatrix, ImplicitDualQuaternion

import pytest

se3_representations = [
    ProjectiveMatrix(),
    ImplicitDualQuaternion(),
]


class TestDifferentiableOpenChainMechanism:
    @pytest.mark.parametrize("se3", se3_representations)
    def test_compute_error_pose_cr5(self, se3):
        urdf_robot = UrdfRobotLibrary.dobot_cr5()
        open_chains = urdf_robot.extract_open_chains(se3, 0.3)
        open_chain = open_chains[-1]
        for _ in range(1000):
            coordinates = []
            for i in range(len(urdf_robot.joint_names)):
                coordinates.append(
                    random.uniform(
                        urdf_robot.joint_limits[i][0], urdf_robot.joint_limits[i][1]
                    )
                )
            coordinates = torch.Tensor(coordinates).unsqueeze(0)
            transformation = open_chain.forward_kinematics(coordinates)
            pose = se3.log(transformation)
            error_pose = open_chain.compute_error_pose(coordinates, pose)
            expected_error_pose = torch.zeros(error_pose.shape)
            assert (error_pose - expected_error_pose).abs().sum() <= 1e-3

    @pytest.mark.parametrize("se3", se3_representations)
    def test_inverse_kinematics_network(self, se3):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        initial = se3.identity(1)
        initial_twist = se3.log(initial)
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial_twist, [(0, 100.0), (0, math.pi * 2)], se3
        )
        thetas = torch.Tensor([[10.0, np.pi / 4]])
        se3_element = open_chain.forward_kinematics(thetas)
        pose = se3.log(se3_element)
        target_pose = pose
        found_thetas = open_chain.inverse_kinematics_backprop(
            initial_thetas=torch.Tensor([[0.0, 0.0]]),
            target_pose=target_pose,
            min_error=1e-2,
            error_weights=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            lr=0.01,
            max_steps=40000,
        )
        assert (found_thetas - thetas).abs().sum() <= 1e-2

    @pytest.mark.parametrize("se3", se3_representations)
    def test_inverse_kinematics_cr5(self, se3):
        urdf_robot = UrdfRobotLibrary.dobot_cr5()
        open_chains = urdf_robot.extract_open_chains(se3, 0.3)
        for _ in range(3):
            coordinates = []
            for i in range(len(urdf_robot.joint_names)):
                coordinates.append(
                    random.uniform(
                        urdf_robot.joint_limits[i][0], urdf_robot.joint_limits[i][1]
                    )
                )
            coordinates = torch.Tensor(coordinates).unsqueeze(0)
            open_chain = open_chains[-1]
            transformation = open_chain.forward_kinematics(coordinates)
            pose = se3.log(transformation)
            target_pose = pose + (torch.rand(6) * 0.001)
            initial_thetas = coordinates
            parameter_update_rate = 0.001 * torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            error_weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            found_thetas = open_chain.inverse_kinematics(
                initial_thetas=initial_thetas,
                target_pose=target_pose,
                min_error=1e-3,
                error_weights=error_weights,
                parameter_update_rate=parameter_update_rate,
                max_steps=100,
            )
            initial_error_pose = open_chain.compute_error_pose(
                initial_thetas, target_pose
            )
            initial_error_pose = (
                DifferentiableOpenChainMechanism.compute_weighted_error(
                    initial_error_pose, error_weights
                )
            )
            found_error_pose = open_chain.compute_error_pose(found_thetas, target_pose)
            found_error_pose = DifferentiableOpenChainMechanism.compute_weighted_error(
                found_error_pose, error_weights
            )
            relative_improvement = found_error_pose / (initial_error_pose + 1e-15)
            """
            At least a 10% improvement w.r.t. initial error
            """
            assert relative_improvement.mean() < 1.0

    @pytest.mark.parametrize("se3", se3_representations)
    def test_inverse_kinematics(self, se3):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        initial = se3.identity(1)
        initial_twist = se3.log(initial)
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial_twist, [(0, 100.0), (0, math.pi * 2)], se3
        )
        thetas = torch.Tensor([[10.0, np.pi / 4]])
        matrix = open_chain.forward_kinematics(thetas)
        pose = se3.log(matrix)

        target_pose = pose
        found_thetas = open_chain.inverse_kinematics(
            initial_thetas=torch.Tensor([[0.0, 0.0]]),
            target_pose=target_pose,
            min_error=1e-2,
            error_weights=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            parameter_update_rate=torch.Tensor([0.5, 0.5]),
            max_steps=10000,
        )
        assert (found_thetas - thetas).abs().sum() <= 1e-2

    def test_compute_weighted_error(self):
        error_twist = torch.Tensor(
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]
        ).float()
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_twist, weights
        )
        (error - torch.Tensor([0.1, 0.4, 0.5])).abs().sum() < 1e-10

    @pytest.mark.parametrize("se3", se3_representations)
    def test_compute_error_pose(self, se3):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        initial = se3.identity(1)
        initial_twist = se3.log(initial)
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial_twist, [(0, 100.0), (0, math.pi * 2)], se3
        )
        target_pose = torch.Tensor([[0, 0, 0, 0, 0, 0]])
        # test zero pose and zero thetas
        thetas = torch.Tensor([[0.0, 0.0]])
        error_twist = open_chain.compute_error_pose(thetas, target_pose)
        assert error_twist.abs().sum() < 1e-10
        # test movement of 10 from identity target
        thetas = torch.Tensor([[10.0, 0.0]])
        error_twist = open_chain.compute_error_pose(thetas, target_pose)
        assert (
            error_twist - torch.Tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
        ).abs().sum() < 1e-7
        # test rotation of 45 deg. from identity
        thetas = torch.Tensor([[0.0, np.pi / 4]])
        error_twist = open_chain.compute_error_pose(thetas, target_pose)
        assert (
            error_twist - torch.Tensor([[0.0, 0.0, 0.0, np.pi / 4, 0.0, 0.0]])
        ).abs().sum() < 1e-7

    @pytest.mark.parametrize("se3", se3_representations)
    def test_jacobian(self, se3):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        - rotate 90 degrees around x and then translating towards y
          ( which is x wrt the original frame)
        - translate 10 meters in z and rotate around x PI rads

        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        thetas = torch.Tensor([[10.0, np.pi], [math.pi / 2.0, 10.0], [10.0, np.pi]])
        initial = torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 10.0],
                [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        se3_projective = ProjectiveMatrix()
        initial_twist = se3_projective.log(to_left_multiplied(initial.unsqueeze(0)))
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial_twist, [(0, 100.0), (0, math.pi * 2)], se3
        )
        jacobian = open_chain.jacobian(thetas)
        """
        Verify size is [ Batch, Twist, Coordinate]
        """
        assert jacobian.shape == torch.Size([3, 6, 2])
        """
        Translation parametes do *not* affect rotation velocities
        """
        rotation_idx = [3, 4, 5]
        translation_idx = [0, 1, 2]
        translation_thetas = (
            torch.Tensor([[1, 0], [0, 1], [1, 0]]).unsqueeze(1).expand(jacobian.shape)
        )
        rotation_jacobian_by_translation_thetas = jacobian[:, rotation_idx, :][
            translation_thetas[:, rotation_idx, :] == 1
        ]
        assert rotation_jacobian_by_translation_thetas.abs().sum() < 1e-10
        """
        Rotation parameters affect rotation velocities.
        """
        rotation_thetas = 1 - translation_thetas
        rotation_jacobian_by_rotation_thetas = jacobian[:, rotation_idx, :][
            rotation_thetas[:, rotation_idx, :] == 1
        ]
        assert rotation_jacobian_by_rotation_thetas.abs().sum() > 0.0
        """
        Rotation parameters affect translation velocities
          - When rotation happens, translation can (and often does) take place.
          (e.g. robotic arms move using rotation parameters to move the robot)
          - SO(3) is a *semi* product: translation parameters do not
          affect angular velocities but rotation parameters do affect
          translation.
        """
        translation_jacobian_by_rotation_thetas = jacobian[:, translation_idx, :][
            rotation_thetas[:, translation_idx, :] == 1
        ]
        assert translation_jacobian_by_rotation_thetas.abs().sum() > 0.0
        """
        Translation parameters affect translation velocities
        """
        translation_jacobian_by_translation_thetas = jacobian[:, translation_idx, :][
            translation_thetas[:, translation_idx, :] == 1
        ]
        assert translation_jacobian_by_translation_thetas.abs().sum() > 0.0
        jacobian_pseudoinverse = torch.linalg.pinv(jacobian)
        velocity_delta = torch.ones([3, 6, 1]) * 0.01
        torch.bmm(jacobian_pseudoinverse, velocity_delta)

    @pytest.mark.parametrize("se3", se3_representations)
    def test_forward_kinematics(self, se3):
        """
        Open Chains:
        - translate 10 meters in z and rotate around x PI rads
        - rotate 90 degrees around x and then translating towards y
          ( which is x wrt the original frame)
        - translate 10 meters in z and rotate around x PI rads

        """
        screws = torch.Tensor(
            [
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            ]
        )
        thetas = torch.Tensor([[10.0, np.pi], [math.pi / 2.0, 10.0], [10.0, np.pi]])
        initial = torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 10.0],
                [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                [0, 0, 0, 1],
            ]
        )
        se3_projective = ProjectiveMatrix()
        initial_twist = se3_projective.log(to_left_multiplied(initial.unsqueeze(0)))
        open_chain = DifferentiableOpenChainMechanism(
            screws, initial_twist, [(0, 100.0), (0, math.pi * 2)], se3
        )
        element = open_chain.forward_kinematics(thetas)
        expected_matrix = torch.Tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 10.0],
                    [0, math.cos(math.pi / 2.0), -math.sin(math.pi / 2.0), 0.0],
                    [0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0), 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 10.0],
                    [0, 0, 0, 1],
                ],
            ]
        )
        for i in range(expected_matrix.shape[0]):
            expected_matrix[i, :, :] = expected_matrix[i, :, :] @ initial[:, :]
        expected_pose = se3_projective.log(to_left_multiplied(expected_matrix))
        computed_pose = se3.log(element)
        assert np.isclose(
            expected_pose,
            computed_pose,
            rtol=1e-05,
            atol=1e-05,
        ).all()


class UrdfRobot:
    @pytest.mark.parametrize("se3", se3_representations)
    def test_extract_open_chains(self, se3):
        se3_projective = ProjectiveMatrix()
        urdf_robot = UrdfRobotLibrary.dobot_cr5()
        open_chains = urdf_robot.extract_open_chains(se3, 0.3)
        for _ in range(100):
            coordinates = []
            for i in range(len(urdf_robot.joint_names)):
                coordinates.append(
                    random.uniform(
                        urdf_robot.joint_limits[i][0], urdf_robot.joint_limits[i][1]
                    )
                )
            coordinates = torch.Tensor(coordinates).unsqueeze(0)
            transformations = urdf_robot.transformations(coordinates)
            for i, transformation in enumerate(transformations):
                computed = open_chains[i].forward_kinematics(coordinates[:, : i + 1])
                computed_pose = se3.log(computed)
                expected = se3_projective.log(
                    to_left_multiplied(torch.Tensor(transformation).unsqueeze(0))
                )
                self.assertTrue(
                    np.isclose(
                        computed_pose,
                        expected,
                        rtol=1e-05,
                        atol=1e-05,
                    ).all()
                )
