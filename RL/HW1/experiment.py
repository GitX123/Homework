import numpy as np
import matplotlib.pyplot as plt
from env import Gaussian_MAB, Bernoulli_MAB
from algo import EpislonGreedy, UCB, Gradient
from utils import plot

num_of_bandits = 50
num_of_eps = 100
num_of_steps = 500

def e_greedy(epsilon = [0, 0.01, 0.1, 0.5, 0.99]):
    
    avg_reward = np.zeros((len(epsilon), num_of_steps)) # average reward of each time step with epsilon e
    for i, e in enumerate(epsilon):
        for _ in range(num_of_eps):
            mab = Gaussian_MAB(num_of_bandits)
            agent = EpislonGreedy(epislon=e)
            for t in range(num_of_steps):
                a = agent.act(t)
                r = mab.step(a)
                avg_reward[i, t] += r / num_of_eps
                agent.update(a, r)
    # figure
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1,1,1)

    for i in range(len(epsilon)):
        ax.plot(range(num_of_steps), avg_reward[i], label='epsilon = {}'.format(epsilon[i]))
    ax.legend()
    ax.set_xlabel("Time step")
    ax.set_ylabel("Average Reward")        
    plt.show()

# run ucb with c = 1~5
def ucb(ucb_c = range(1, 6)):
    avg_reward = np.zeros((len(ucb_c), num_of_steps)) # average reward of each time step with epsilon e

    for i, c in enumerate(ucb_c):
        for _ in range(num_of_eps):
            mab = Gaussian_MAB(num_of_bandits)
            agent = UCB(c=c)
            for t in range(num_of_steps):
                a = agent.act(t)
                r = mab.step(a)
                avg_reward[i, t] += r / num_of_eps
                agent.update(a, r)
    
    # figure
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(1,1,1)

    for i in range(len(ucb_c)):
        ax.plot(range(num_of_steps), avg_reward[i], label='c = {}'.format(ucb_c[i]))
    ax.legend()
    ax.set_xlabel("Time step")
    ax.set_ylabel("Average Reward")        
    plt.show()

# run all experiments with number of bandits = 10, 20, 30, 40, 50
def runAll(num_of_bandits = range(10, 51, 10)):
    mabs = list(Gaussian_MAB(nb) for nb in num_of_bandits)
    avg_reward = np.zeros((3, len(num_of_bandits), num_of_steps))

    for i, nb in enumerate(num_of_bandits):
        for _ in range(num_of_eps):
            mab = Gaussian_MAB(nb)
            agents = [EpislonGreedy(NumofBandits=nb), UCB(NumofBandits=nb), Gradient(NumofBandits=nb)]
            r_avg = 0
            for t in range(num_of_steps):
                # action
                a_e = agents[0].act(t)
                a_ucb = agents[1].act(t)
                a_grad = agents[2].act(t)
                # reward
                r_e = mab.step(a_e)
                r_ucb = mab.step(a_ucb)
                r_grad = mab.step(a_grad)
                r_avg += (r_grad - r_avg) / (t + 1)
                # update
                agents[0].update(a_e, r_e)
                agents[1].update(a_ucb, r_ucb)
                agents[2].update(a_grad, r_grad, r_avg)
                # store step reward
                avg_reward[0, i, t] += r_e / num_of_eps
                avg_reward[1, i, t] += r_ucb / num_of_eps
                avg_reward[2, i, t] += r_grad / num_of_eps
    
    fig = plt.figure(figsize=[30, 5])
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for i, nb in enumerate(num_of_bandits):
        ax1.plot(range(num_of_steps), avg_reward[0, i], label='{} bandits'.format(nb))
    for i, nb in enumerate(num_of_bandits):
        ax2.plot(range(num_of_steps), avg_reward[1, i], label='{} bandits'.format(nb))
    for i, nb in enumerate(num_of_bandits):
        ax3.plot(range(num_of_steps), avg_reward[2, i], label='{} bandits'.format(nb))

    ax1.legend()
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Average Reward")        
    ax2.legend()
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Average Reward")     
    ax3.legend()
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Average Reward")     
    plt.show()

if __name__ == '__main__':
    # e_greedy()
    ucb()
    # runAll()