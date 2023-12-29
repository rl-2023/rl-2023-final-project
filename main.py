import gym

# need to run the pressure plate import to register the environment with gym
import pressureplate

if __name__ == "__main__":
    env = gym.make('pressureplate-linear-4p-v0')

    env.reset()

    # all agents go up
    env.step([0, 0, 0, 0])