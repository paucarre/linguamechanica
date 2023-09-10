import pybullet as p
from time import sleep

from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary
import click
import logging


def get_logger():
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger("ik_test")
    return logger


@click.command(context_settings = dict( help_option_names = ['-h', '--help'] ))
@click.option("--checkpoint", help="Model checkpoint identifier.", type=int, required=True)
@click.option("--urdf", default="./urdf/cr5.urdf", help="URDF of the robot.", type=str, required=False)
def test(checkpoint, urdf):
    print(checkpoint)
    logger = get_logger()
    # setup pybullet
    p.connect(p.GUI)
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
    num_joints = p.getNumJoints(robot_id)
    # load agent and environment
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    agent.training_state.episode_batch_size = 1
    environment = Environment(
        open_chain=open_chain, training_state=agent.training_state
    ).cuda()
    # set initial state
    state, initial_reward = environment.reset()
    thetas, target_pose = Environment.thetas_target_pose_from_state(state)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, thetas[0, i].item())
        p.resetJointState(initial_robot_id, i, thetas[0, i].item())
        p.resetJointState(target_robot_id, i, environment.target_thetas[0, i].item())

    # start simulation
    finished = False
    while not finished:
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        action_mean, actions, log_probabilities, entropy = agent.choose_action(
            state, agent.training_state
        )
        actions, next_state, reward, done = environment.step(actions)
        for i in range(num_joints):
            p.resetJointState(robot_id, i, thetas[0, i].item())
        sleep(0.1)
        state = next_state
        logger.info(
            f"Initial Reward: {initial_reward.item():1.3} | Current Reward {reward.item():1.3} "
        )


if __name__ == "__main__":
    test()
