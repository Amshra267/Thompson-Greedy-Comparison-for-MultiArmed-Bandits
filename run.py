from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Agents import Bandit, EpsilonGreedyAgent, ThomspsonSamplingAgent
import pandas as pd


def compare_epsilons_and_TS(
    epsilons: List[float],
    bandits_true_means: List[float],
    timesteps: int,
    num_simulations:int):
    """
    Compare different epsilons for epsilon-greedy algorithm over num_simulations
    """
    bandits = [Bandit(id, m) for id, m in enumerate(bandits_true_means)]
    
    Agents_rewards = np.zeros((len(epsilons), timesteps))
    Agents_actions = np.zeros((len(epsilons), len(bandits_true_means), timesteps))
    _, ax1 = plt.subplots()
    for n in range(num_simulations):
        for ag, epsilon in enumerate(epsilons):
            if epsilon == "TS":
                print("Running TS for simulation_num  = {}".format(n+1))
                agent = ThomspsonSamplingAgent(bandits=bandits)
            else:
                print("Running epsilonwith epsilon = {} for simulation_num  = {}".format(epsilon, n+1))
                agent = EpsilonGreedyAgent(bandits=bandits, epsilon=epsilon)
            #   print(timesteps)
            rewards, actions = agent.actions(timesteps)
            Agents_rewards[ag]+=rewards  # aadding rewards for averaging
            ax1.plot(np.divide(Agents_rewards[ag], n+1), label = "epsilon = " + str(epsilon) if epsilon !="TS" else "TS")
            ax1.legend()
            for j in range(len(bandits_true_means)):  # loop over all actions to find the Average percentage of each action at a timestep
                Agents_actions[ag][j] += np.uint8(np.array(actions)==j)
        
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("Average_Expected_reward")
        plt.savefig("Comparisons")
        ax1.clear()
        
    return Agents_actions

epsilons = [0, 0.01, 0.1, "TS"]
bandits_means = list(np.random.randn(10)) # 10 bandits sampled from standard normal distribution

# Finding the true optimal action of a particular run with it expected value for each action
print("True Order of Optimal Actions is")
ranked = np. argsort(bandits_means)
sort_indices = ranked[::-1]
print("Ranks -", sort_indices)
print("corresponding_values = ", np.array(bandits_means)[sort_indices])
print("-"*50)

# plotting true reward functions with Q*(a) as bandit_means
rewards = []

for i, b_m in enumerate(bandits_means):
    for _ in range(1000):
        rewards.append([str(i), np.random.randn() + b_m])

data = pd.DataFrame(rewards, columns=['Actions', 'Rewards'])
sns.violinplot(data = data, x ="Actions", y = "Rewards")
plt.title("True Reward Distribution")
plt.savefig("True_Rewards.png")


# Now comes the Actual code part

if __name__ == "__main__":
    timesteps = 1500
    num_simulations = 2500
    Agents_actions = compare_epsilons_and_TS(epsilons, bandits_means, timesteps, num_simulations)
    fig, (ax1, ax2) = plt.subplots(figsize = (10,5), ncols= 2) 
    for i in range(len(bandits_means)):
        ax1.plot(np.divide(Agents_actions[1][i], num_simulations)*100, label = "Action = " + str(i))
        ax2.plot(np.divide(Agents_actions[3][i], num_simulations)*100, label = "Action = " + str(i))
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Average Optimal Action Percentage")
    ax1.set_title("For Greedy agent with epsilon = 0.01")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Average Optimal Action Percentage")
    ax2.set_title("For TS")
    plt.savefig("Optimal_Actions.png")

    print("Ranks = ", sort_indices)  # Again printing in the last to see