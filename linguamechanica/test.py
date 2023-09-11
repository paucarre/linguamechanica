import pybullet as p
from dataclasses import dataclass

from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary
import click
import logging
import torch


@dataclass
class PyBulletRobotIds:
    robot_id: int
    target_robot_id: int
    initial_robot_id: int


def get_logger():
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger("ik_test")
    logger.setLevel(logging.INFO)
    return logger


def setup_pybullet(urdf):
    get_logger()
    # setup pybullet
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # load robot
    robot_id = p.loadURDF(urdf, [0, 0, 0])
    target_robot_id = p.loadURDF(urdf, [0, 0, 0])
    initial_robot_id = p.loadURDF(urdf, [0, 0, 0])
    for link in range(-1, 20):
        p.changeVisualShape(target_robot_id, link, rgbaColor=[0.1, 0.8, 0.4, 0.5])
        p.changeVisualShape(robot_id, link, rgbaColor=[0.1, 0.1, 0.8, 0.5])
        p.changeVisualShape(initial_robot_id, link, rgbaColor=[0.5, 0.5, 0.5, 0.5])
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(target_robot_id, [0, 0, 0], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(initial_robot_id, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, 0)
    return PyBulletRobotIds(robot_id, target_robot_id, initial_robot_id)


def setup_inference(robot_ids, urdf, checkpoint, samples, level):
    # load agent and environment
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    agent.training_state.episode_batch_size = samples
    agent.training_state.level = level
    environment = Environment(
        open_chain=open_chain, training_state=agent.training_state
    ).cuda()
    # set initial state
    target_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state, initial_reward = environment.reset_to_target_pose(target_pose)
    # TODO: this is a nasty hack
    # environment.target_pose = state[0:1, :6]
    # TODO: this is a nasty hack
    # force target pose to be the same
    # state[:, :6] = state[0:1, :6]
    thetas, target_pose = Environment.thetas_target_pose_from_state(state)
    for i in range(p.getNumJoints(robot_ids.robot_id)):
        p.resetJointState(robot_ids.robot_id, i, thetas[0, i].item())
        p.resetJointState(robot_ids.initial_robot_id, i, thetas[0, i].item())
        p.resetJointState(
            robot_ids.target_robot_id, i, environment.target_thetas[0, i].item()
        )
    return environment, agent, state


def solve_ik(robot_ids, iterations, state, agent, environment):
    iteration = 0
    max_reward = None
    while iteration < iterations:
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        # thetas = thetas.cuda()
        action_mean, actions, log_probabilities, entropy = agent.choose_action(
            state, agent.training_state
        )
        actions, next_state, reward, done = environment.step(actions)
        current_best_reward_idx = torch.argmax(reward, dim=0)
        current_best_reward = reward[current_best_reward_idx, 0].item()
        if max_reward is None or max_reward < current_best_reward:
            max_reward = current_best_reward
            max_thetas = thetas[current_best_reward_idx.to(thetas.device), :]
            for i in range(p.getNumJoints(robot_ids.robot_id)):
                p.resetJointState(robot_ids.robot_id, i, max_thetas[0, i].item())
        state = next_state
        iteration += 1
    while True:
        pass


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--checkpoint", help="Model checkpoint identifier.", type=int, required=True
)
@click.option(
    "--level",
    help="IK Game Level (theta noise is '0.1 * level').",
    type=int,
    default=3,
    required=True,
)
@click.option(
    "--urdf",
    default="./urdf/cr5.urdf",
    help="URDF of the robot.",
    type=str,
    required=False,
)
@click.option(
    "--samples",
    default=1024,
    help="Number of initial poses to solve the IK problem.",
    type=int,
    required=False,
)
@click.option(
    "--iterations",
    default=100,
    help="Number of IK iterations.",
    type=int,
    required=False,
)
def test(checkpoint, urdf, level, samples, iterations):
    robot_ids = setup_pybullet(urdf)
    environment, agent, state = setup_inference(
        robot_ids, urdf, checkpoint, samples, level
    )
    solve_ik(robot_ids, iterations, state, agent, environment)


if __name__ == "__main__":
    test()
