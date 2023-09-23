import click
import torch

from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary


def target_thetas_reset(environment, target_thetas):
    target_thetas = [float(theta) for theta in target_thetas.split(",")]
    target_thetas = torch.tensor(target_thetas)
    return environment.reset_to_target_thetas(target_thetas)


def target_pose_reset(environment, target_pose):
    target_pose = [float(element) for element in target_pose.split(",")]
    target_pose = torch.tensor(target_pose)
    return environment.reset_to_target_pose(target_pose)


def setup_inference(urdf, checkpoint, samples, target_thetas, target_pose):
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    # TODO: make this generic
    se3 = ProjectiveMatrix()
    open_chain = urdf_robot.extract_open_chains(se3, 0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    agent.training_state.episode_batch_size = samples
    environment = Environment(
        open_chain=open_chain, training_state=agent.training_state
    ).cuda()
    state, initial_reward = None, None
    if target_thetas is not None:
        state, initial_reward = target_thetas_reset(environment, target_thetas)
    elif target_pose is not None:
        state, initial_reward = target_pose_reset(environment, target_pose)
    thetas, target_pose = Environment.thetas_target_pose_from_state(state)
    return environment, agent, state, initial_reward


def inference_results_to_csv(thetas_sorted, reward_sorted):
    thetas_and_rewards_sorted = torch.cat([thetas_sorted, reward_sorted], 1)
    headers = [f"theta_{idx + 1}" for idx in range(thetas_sorted.shape[1])] + ["reward"]
    thetas_and_rewards_sorted = [headers] + thetas_and_rewards_sorted.tolist()
    thetas_and_rewards_sorted = "\n".join(
        [
            ",".join([str(theta) for theta in current_thetas])
            for current_thetas in thetas_and_rewards_sorted
        ]
    )
    return thetas_and_rewards_sorted


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--checkpoint", help="Model checkpoint identifier.", type=int, required=True
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
@click.option("--target_thetas", type=str, required=False)
@click.option("--target_pose", type=str, required=False)
@click.option("--top_n", type=int, default=10, required=True)
def inference(checkpoint, urdf, samples, iterations, target_thetas, target_pose, top_n):
    target_thetas is not None
    environment, agent, state, initial_reward = setup_inference(
        urdf, checkpoint, samples, target_thetas, target_pose
    )
    thetas_sorted, reward_sorted = agent.inference(
        iterations, state, environment, top_n
    )
    inference_csv = inference_results_to_csv(thetas_sorted, reward_sorted)
    print(inference_csv)


if __name__ == "__main__":
    inference()
