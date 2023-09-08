import torch
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from torch.utils.tensorboard import SummaryWriter
from linguamechanica.training_context import EpisodeState, TrainingState


def evaluate_policy(agent, training_state, summary):
    environment = Environment(
        training_state.episode_batch_size, agent.open_chain, training_state
    )
    state = environment.reset()
    episode = EpisodeState("Test", training_state.gamma)
    finished = False
    while not finished:
        action, log_prob, mu_v, noise = agent.choose_action(state, training_state)
        action, next_state, reward, done = environment.step(action)
        finished = episode.step(reward, done, training_state.t, summary)


def train():
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    # TODO: do it well!
    a = torch.zeros(1).cuda()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].to(a.device)
    # TODO: place all these constants as arguments

    training_state = TrainingState()
    env = Environment(training_state.episode_batch_size, open_chain, training_state).to(
        a.device
    )
    summary = SummaryWriter()
    agent = IKAgent(
        open_chain=open_chain,
        summary=summary,
        lr_actor=training_state.lr_actor,
        lr_actor_geodesic=training_state.lr_actor_geodesic,
        lr_critic=training_state.lr_critic,
        state_dims=(env.state_dimensions),
        action_dims=env.action_dims,
        gamma=training_state.gamma,
        policy_freq=training_state.policy_freq,
        tau=training_state.tau,
        max_action=training_state.max_action,
        initial_action_variance=training_state.initial_action_variance,
        max_variance=training_state.max_variance,
        replay_buffer_max_size=training_state.replay_buffer_max_size(),
    )
    episode = EpisodeState("Train", training_state.gamma)
    state = env.reset()
    # agent.load(f"checkpoint_46999")

    for training_state.t in range(training_state.t, int(training_state.max_time_steps)):
        action, log_prob, mu_v, noise = agent.choose_action(
            state, training_state, summary
        )
        action, next_state, reward, done = env.step(action)
        summary.add_scalar("Loss / Train / Step Batch Reward", reward.mean(), training_state.t)
        agent.store_transition(
            state=state.detach().cpu(),
            action=action.detach().cpu(),
            reward=reward.detach().cpu(),
            next_state=next_state.detach().cpu(),
            done=done.detach().cpu(),
        )
        if episode.step(reward, done, training_state.t, summary):
            state = env.reset()
        else:
            state = next_state
        if training_state.can_train_buffer():
            agent.train_buffer(training_state)
        if training_state.can_evaluate_policy():
            evaluate_policy(agent, training_state, summary)
        if training_state.can_save():
            agent.save(training_state)

    summary.close()


if __name__ == "__main__":
    train()
