"""On-policy tabular SARSA implemented from scratch."""
import numpy as np

class SarsaAgent:
    def __init__(self, n_states=48, n_actions=4, alpha=0.5, gamma=1.0, seed=30):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, next_action, terminated):
        # SARSA uses the next action selected by the behavior policy, not max Q.
        target = reward if terminated else reward + self.gamma * self.q[next_state, next_action]
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def save(self, path):
        np.save(path, self.q)
