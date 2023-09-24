import matplotlib.pyplot as plt

# /content/ReLLIE/State_de.py:70: RuntimeWarning: invalid value encountered in divide
#   nsigma = 0 + (np.max(nsigma)*2 - 0) * (nsigma - np.min(nsigma)) / (np.max(nsigma) - np.min(nsigma))
plt.switch_backend('TkAgg')
def plot_rewards_from_txt(txt_file):
    episode_rewards = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        episode_reward = 0  # Initialize the episode reward
        for line in lines:
            if "total reward" in line:
                reward_line = line.strip().split()
                episode_reward += round(float(reward_line[-1]))  # Accumulate reward
            elif "episode" in line:
                episode_rewards.append(episode_reward)
                episode_reward = 0  # Reset episode reward at the start of a new episode

    episodes = list(range(1, len(episode_rewards) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, marker='o', linestyle='-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

# Example usage:
txt_file = 'res_ex6.txt'
plot_rewards_from_txt(txt_file)
