import gymnasium as gym
import itertools
import agent_class as agent
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("plots", exist_ok=True)
os.makedirs("videos", exist_ok=True)


env = gym.make('LunarLander-v3')
observation, info = env.reset()
N_state = len(observation)
N_actions = env.action_space.n

print(f"State space dimension: {N_state}")
print(f"Number of actions: {N_actions}")


def train_and_plot(agent_instance, algo_name):
    print(f"\n=== Training {algo_name.upper()} ===")
    results = agent_instance.train(environment=env, verbose=True)
    returns = results['epsiode_returns']
    durations = results['episode_durations']

    # Compute running mean
    N = 20
    def running_mean(x, N=20):
        x_out = np.zeros(len(x)-N)
        for i in range(len(x)-N):
            x_out[i] = np.mean(x[i:i+N])
        return x_out

    # === Plot and save ===
    plt.figure(figsize=(6,4))
    plt.plot(returns, label='Episode Return')
    plt.plot(np.arange(N, len(returns)), running_mean(returns), label='Running Mean')
    plt.title(f'{algo_name.upper()} Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()

    # Save plot instead of showing
    plot_path = os.path.join("plots", f"{algo_name}_training.png")
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"📊 Saved training plot for {algo_name} → {plot_path}")

    return results


# === Train and compare all three algorithms ===

agents = {}

# DQN
params_dqn = {'N_state': N_state, 'N_actions': N_actions}
agents['DQN'] = agent.dqn(parameters=params_dqn)

# Double DQN
params_ddqn = {'N_state': N_state, 'N_actions': N_actions, 'doubleDQN': True}
agents['DDQN'] = agent.dqn(parameters=params_ddqn)

# Actor-Critic
params_ac = {'N_state': N_state, 'N_actions': N_actions}
agents['Actor-Critic'] = agent.actor_critic(parameters=params_ac)

# === Train all ===
training_results = {}
for name, ag in agents.items():
    training_results[name] = train_and_plot(ag, name)


# === Record videos for all trained agents ===
from gymnasium.wrappers import RecordVideo

for name, ag in agents.items():
    print(f"\n🎥 Generating video for {name} agent...")

    # Wrap the environment with RecordVideo
    env = gym.make('LunarLander-v3', render_mode="rgb_array")
    env = RecordVideo(env, video_folder=f"./videos/{name}", episode_trigger=lambda e: True)

    state, info = env.reset()
    total_reward = 0
    for t in itertools.count():
        action = ag.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"{name} finished with total reward = {total_reward:.2f}")
            break

    env.close()

print("\n✅ Training complete! Plots saved in 'plots' and videos saved in 'videos'.")
