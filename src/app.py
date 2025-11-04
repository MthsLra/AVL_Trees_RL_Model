import gymnasium as gym 

env_name = "AVL_tree"
env = gym.make(env_name, options={})
observation = env.reset()


done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode="human")


env.close()
