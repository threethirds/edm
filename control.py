from env.edm import NumpyEDM1 as EDM
from agent.edm import Network as Agent
from env.edm import ReturnTracker
from env.edm import TorchEnvWrapper
from env.edm import VectorEnv

env = TorchEnvWrapper(VectorEnv([ReturnTracker(EDM(1.), max_steps=1000) for _ in range(4)]), 'cpu')
obs = env.reset()

agent = Agent(prob_uniform=0.)
agent.eval()


for step in range(100):

    value, a_dist = agent(obs)
    a = a_dist.sample()
    obs, _, _, _ = env.step(a)

    print(step, a, obs)
