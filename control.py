import torch

from env.edm import NumpyEDM1 as EDM
from agent.edm import Network as Agent

env = EDM(1.)
obs = env.reset()

agent = Agent(prob_uniform=0.)
agent.load_state_dict(torch.load('agent_weights.pt'))
agent.eval()

for step in range(100):

    value, a_dist = agent(torch.from_numpy(obs).to(torch.float32).unsqueeze(0))
    a = a_dist.sample()
    obs, _, _, _ = env.step(a)

    print(step, a, obs)
