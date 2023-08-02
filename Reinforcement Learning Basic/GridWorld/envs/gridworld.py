class GridWorldEnv:
    def __init__(self, gridSize=7, startState='00', terminalStates=None, ditches=None, ditchPenalty=-10,
                 turnPenalty=-1, winReward=100, mode='prod'):
        """
        Create an instance of grid world environment.
        Parameters
        :param gridSize: int
            The size of grid
        :param startState: str
            Starting state of the environment e.g) '01'
        :param terminalStates: list(str)
            States that terminate the environment and return a reward of :param winReward
        :param ditches: list(str)
            Penalty states that return a reward of :param ditchPenalty
        :param ditchPenalty: int
        :param turnPenalty:
        :param winReward: int
        :param mod: str
            prod/debug
        """
        if ditches is None:
            ditches = ['52']
        if terminalStates is None:
            terminalStates = ['64']
        self.state_space = []
        self.mode = mode
        self.winReward = winReward
        self.turnPenalty = turnPenalty
        self.ditchPenalty = ditchPenalty
        self.ditches = ditches
        self.terminalStates = terminalStates
        self.gridSize = min(9, gridSize)
        self.startState = startState

        self.action_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.action_space = [0, 1, 2, 3]
        self.create_statespace()
        self.state_count = self.get_statespace_len()
        self.action_count = self.get_action_space()
        self.state_dict = {key: value for key, value in zip(self.state_space, range(self.state_count))}
        self.current_state = self.startState
        self.game_ended = False
        self.total_turns = 0
        if self.mode == 'debug':
            print("State Space", self.state_space)
            print("State Dict", self.state_dict)
            print("Action Space", self.action_space)
            print("Action Dict", self.action_dict)
            print("Start State", self.startState)
            print("Terminal States", self.terminalStates)
            print("Ditches", self.ditches)
            print("WinReward:{}, TurnPenalty:{}, DitchPenalty:{}"
                  .format(self.winReward, self.turnPenalty, self.ditchPenalty))

    def create_statespace(self):
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                self.state_space.append(str(i) + str(j))

    def set_mode(self, mode):
        self.mode = mode

    def get_statespace(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    def get_action_dict(self):
        return self.action_dict

    def get_statespace_len(self):
        return len(self.state_space)

    def get_action_space_len(self):
        return len(self.action_space)

    def next_state(self, current_state, action):
        row = int(current_state[0])
        col = int(current_state[1])
        if action == 0:
            row = max(0, row - 1)
        if action == 1:
            row = min(self.gridSize - 1, row + 1)
        if action == 2:
            col = max(0, col - 1)
        if action == 3:
            col = min(self.gridSize - 1, col + 1)
        new_state = str(row) + str(col)
        if new_state in self.state_space:
            if new_state in self.terminalStates:
                self.game_ended = True
            if self.mode == 'debug':
                print("CurrentState:{}, Action:{}, NextState:{}"
                      .format(current_state, action, new_state))
            return new_state
        else:
            return current_state

    def compute_reward(self, state):
        reward = 0
        reward += self.turnPenalty
        if state in self.ditches:
            reward += self.ditchPenalty
        if state in self.terminalStates:
            reward += self.winReward
        return reward

    def reset(self):
        self.game_ended = False
        self.current_state = self.startState
        self.total_turns = 0
        return self.current_state

    def step(self, action):
        if self.game_ended:
            raise "Game is Over Exception"
        if action not in self.action_space:
            raise "Action Invalid Exception"
        self.current_state = self.next_state(self.current_state, action)
        reward = self.compute_reward(self.current_state)
        done = self.game_ended
        self.total_turns += 1
        if self.mode == 'debug':
            print("Obs:{}, Reward:{}, Done:{}, TotalTurns:{}"
                  .format(self.current_state, reward, done, self.total_turns))
        return self.current_state, reward, done, self.total_turns


if __name__ == '__main__':
    env = GridWorldEnv(mode='debug')
    print("Resting Env...")
    env.reset()
    print("Go DOWN...")
    env.step(1)
    print("Go RIGHT...")
    env.step(3)
    print("Go LEFT...")
    env.step(2)
    print("Go UP...")
    env.step(0)
    print("Invalid ACTION...")
    # env.step(4)
