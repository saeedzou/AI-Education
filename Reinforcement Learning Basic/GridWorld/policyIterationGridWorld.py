from envs.gridworld import GridWorldEnv
import numpy as np


class PolicyIteration:
    def __init__(self, env=GridWorldEnv(), discount_factor=0.9, convergence_threshold=1e-4, max_iters_value=1000,
                 max_iters_policy=100, mode='prod'):
        self.gamma = discount_factor
        self.th = convergence_threshold
        self.max_iters_value = max_iters_value
        self.max_iters_policy = max_iters_policy
        self.mode = mode
        self.env = env
        self.action_count = self.env.get_action_space_len()
        self.state_count = self.env.get_statespace_len()
        self.action_dict = self.env.action_dict
        self.state_dict = self.env.state_dict
        self.V = np.zeros(self.state_count)
        self.Q = [np.zeros(self.action_count) for s in range(self.state_count)]
        self.Policy = np.zeros(self.state_count)
        self.total_reward = 0
        self.total_steps = 0

    def compute_value_under_policy(self):
        self.V = np.zeros(self.state_count)
        for i in range(self.max_iters_value):
            V_prev = np.copy(self.V)
            for s in range(self.state_count):
                current_state = self.env.state_space[s]
                for a in self.env.action_space:
                    next_state = self.env.next_state(current_state, a)
                    reward = self.env.compute_reward(next_state)
                    next_state_index = self.env.state_dict[next_state]
                    self.Q[s][a] = reward + self.gamma * V_prev[next_state_index]
                if self.mode == 'debug':
                    print("Q(s={}):{}".format(current_state, self.Q[s]))
                self.V[s] = max(self.Q[s])
            if np.sum(np.fabs(V_prev - self.V)) <= self.th:
                print("Convergence Achieved in {}th iteration. "
                      "Breaking V_Iteration loop!".format(i))
                break

    def reset_episode(self):
        self.total_reward = 0
        self.total_steps = 0

    def iterate_policy(self):
        self.Policy = [np.random.choice(self.action_count) for s in range(self.state_count)]
        for i in range(self.max_iters_policy):
            self.compute_value_under_policy()
            policy_prev = self.Policy
            self.extract_optimal_policy()
            policy_new = self.Policy
            if np.all(policy_new == policy_new):
                print("Policy Converged in step ", i)
                break
            self.Policy = policy_new
        return self.Policy

    def extract_optimal_policy(self):
        self.Policy = np.argmax(self.Q, axis=1)
        if self.mode == 'debug':
            print("Optimal Policy: ", self.Policy)

    def run_episode(self):
        """
        Runs a new episode
        :return: float
            Total Return of episode
        """
        self.reset_episode()
        current_state = self.env.reset()
        while True:
            a = self.Policy[self.state_dict[current_state]]
            new_state, reward, done, _ = self.env.step(a)
            if self.mode == 'debug':
                print(
                    f"Current state: {current_state}, action: {a}, new state: {new_state}, reward: {reward}, done: {done}")
            self.total_reward += reward
            self.total_steps += 1
            if done:
                break
            else:
                current_state = new_state
        return self.total_reward

    def evaluate_policy(self, n_episodes):
        episode_scores = []
        if self.mode == 'debug':
            print("Running {} episodes!".format(n_episodes))
        for e, episode in enumerate(range(n_episodes)):
            score = self.run_episode()
            episode_scores.append(score)
            if self.mode == 'debug':
                print("Score in {} episode = {}".format(e, score))
        return np.mean(episode_scores)

    def solve_mdp(self, n_episode=100):
        if self.mode == 'debug':
            print("Iterating Values...")
        self.iterate_policy()
        if self.mode == 'debug':
            print("Extracting Optimal Policy...")
            self.extract_optimal_policy()
        if self.mode == 'debug':
            print("Scoring Policy...")
        return self.evaluate_policy(n_episode)


if __name__ == '__main__':
    print("Initializing variables and setting environment...")
    policyIteration = PolicyIteration(env=GridWorldEnv(), mode='debug')
    print('Policy Evaluation Score = ', policyIteration.solve_mdp())
