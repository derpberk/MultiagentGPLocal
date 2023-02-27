import numpy as np


class TimeStackingMemory:

    def __init__(self, n_timesteps: int, state_indexes: tuple, n_channels: int):
        
        # Number of timesteps to store
        self.n_timesteps = n_timesteps
        # State index to accumulate
        self.state_indexes = state_indexes
        # Expanded state dictionary
        self.state_memory = {}
        # Number of channels per state #
        self.n_channels = n_channels
        # init flag
        self.init = False


    def process_stacked_space(self, non_stacked_state):
        

        
        if not self.init:

            for i in range(self.n_channels):
                if i in self.state_indexes:
                    self.state_memory[i] = np.repeat(non_stacked_state[i][np.newaxis], self.n_timesteps + 1, axis=0)
                else:
                    self.state_memory[i] = non_stacked_state[i][np.newaxis]

            self.init = True

        else:

            for i in range(self.n_channels):

                if i in self.state_indexes:
                    new_channel_shifted = np.roll(self.state_memory[i], shift=1, axis=0)
                    new_channel_shifted[0] = non_stacked_state[i]
                    self.state_memory[i] = new_channel_shifted
                else:
                    self.state_memory[i] = non_stacked_state[i][np.newaxis]

        return np.concatenate([channel for channel in self.state_memory.values()], axis=0)


class MultiAgentTimeStackingMemory:

    def __init__(self, n_agents: int, n_timesteps: int, state_indexes: tuple, n_channels: int):

        self.memories = [TimeStackingMemory(n_timesteps, state_indexes, n_channels) for _ in range(n_agents)]
        
    def process(self, multiagent_state: dict):

        stacked_state = {}
        for agent_id in multiagent_state.keys():

            stacked_state[agent_id] = self.memories[agent_id].process_stacked_space(multiagent_state[agent_id])

        return stacked_state


if __name__ == '__main__':

    from time import sleep
    time_memory = TimeStackingMemory(n_timesteps = 3, state_indexes = (0,2), n_channels = 3)


    for t in range(10):

        s = np.random.randint(0,10, size=(3,3,3))

        print("He introducido el estado: ")
        print(s)
        exp_s = time_memory.process_stacked_space(s)
        print("El resultado ha sido: ")
        print(exp_s)
