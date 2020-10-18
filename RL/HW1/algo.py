import numpy as np

class EpislonGreedy(object):
    def __init__(self, NumofBandits=10, epislon=0.1):
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        self._Q[action] += (immi_reward - self._Q[action]) / self._action_N[action]

    def act(self, t):
        if np.random.random() > self._epislon:
            max_q = np.amax(self._Q)
            choices = []
            for i in range(self._nb):
                if self._Q[i] == max_q:
                    choices.append(i)
            action = np.random.choice(choices)
        else:
            action = np.random.randint(self._nb)
        
        self._action_N[action] += 1
        return action

class UCB(object):
    def __init__(self, NumofBandits=10, c=2):
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        self._Q[action] += (immi_reward - self._Q[action]) / self._action_N[action]

    def act(self, t):
        choices = []
        for i, n in enumerate(self._action_N):
            if n == 0:
                choices.append(i)

        # if there are actions never taken before
        if choices:
            action = np.random.choice(choices)
        else:
            Q_ucb = np.zeros_like(self._Q)
            for i in range(self._nb):
                Q_ucb[i] = self._Q[i] + self._c * np.sqrt(np.log(t) / self._action_N[i])
            
            max_q = np.amax(Q_ucb)
            for i in range(self._nb):
                if Q_ucb[i] == max_q:
                    choices.append(i)
            action = np.random.choice(choices)
        
        self._action_N[action] += 1
        return action

class Gradient(object):
    def __init__(self, NumofBandits=10, alpha=0.1):
        self._nb = NumofBandits
        self._H = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self._alpha = alpha

    def update(self, action, immi_reward, avg_reward):
        p_a = np.exp(self._H[action]) / np.sum(np.exp(self._H))
        for i in range(self._nb):
            if i != action:
                self._H[i] -= self._alpha * (immi_reward - avg_reward) * p_a
        self._H[action] += self._alpha * (immi_reward - avg_reward) * (1 - p_a)

    def act(self, t):
        p = np.zeros(self._nb) # probabilities
        for i in range(self._nb):
            p[i] = np.exp(self._H[i]) / np.sum(np.exp(self._H))
        action = np.random.choice(self._nb, p=p)
        
        self._action_N[action] += 1
        return action 