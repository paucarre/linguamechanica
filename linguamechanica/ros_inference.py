import click
import torch

from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.se3 import ImplicitDualQuaternion


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
def ros_inference_serialization(checkpoint, urdf):
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    se3 = ImplicitDualQuaternion()
    open_chain = urdf_robot.extract_open_chains(se3, 0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    current_thetas = torch.rand(1, open_chain.dof()).cuda()
    target_pose = torch.rand(1, 6).cuda()
    traced_script_module = torch.jit.trace(agent.actor, (current_thetas, target_pose))
    print(traced_script_module)


if __name__ == "__main__":
    ros_inference_serialization()
