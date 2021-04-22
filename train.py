"""
Notation and implementation follows "Proximal Policy Optimization Algorithms"
by Schulman and others.
"""

import mmflow
import numpy as np
import torch
from pathlib import Path
from tqdm import trange
import yaml
import gym
import mlflow
from mmflow import MlflowLogger
from mmflow.logging.aggregator import ConstantAggregator

from env.edm import NumpyEDM1
from agent.edm import Network
from env.edm import ReturnTracker
from env.edm import TorchEnvWrapper
from env.edm import VectorEnv

global_step = 0


def run_edm_ppo(config_yml: Path, run_name='experiment', user='unknown'):
    parameters = yaml.load(config_yml.open('r'), Loader=yaml.SafeLoader)

    mlflow.set_tracking_uri('http://mlflow-service.default.svc.cluster.local/')

    torch.manual_seed(2)
    n_envs = parameters.get('n_envs', 16)
    area = parameters.get('area', 1)
    cuda = bool(parameters.get('cuda', False))
    device = torch.device('cuda' if cuda else 'cpu')
    ent_coef = parameters.get('ent_coef', 1e-3)
    clip_coef = parameters.get('clip_coef', 0.2)
    vf_coef = parameters.get('vf_coef', 1)
    gamma = parameters.get('gamma', 0.99)
    lr = parameters.get('lr', 3e-4)
    T = parameters.get('max_steps', 1000)
    n_episodes = parameters.get('n_episodes', 10_000)
    update_epochs = parameters.get('update_epochs', 8)
    minibatch_size = parameters.get('minibatch_size', 64)
    prob_uniform = parameters.get('prob_uniform', 0.01)
    flush_penalty = parameters.get('flush_penalty', 0)
    logger = MlflowLogger('edm', config_yml.stem, run_name,
                          tags={'mlflow.user': user},
                          aggregator=ConstantAggregator(T * n_envs * n_episodes / 1000))
    for param_name, param_value in parameters.items():
        if not isinstance(param_value, dict):
            logger.log_param(param_name, param_value)
    _envs = [ReturnTracker(NumpyEDM1(area, flush_penalty=flush_penalty), logger, max_steps=T) for _ in range(n_envs)]
    env = TorchEnvWrapper(VectorEnv(_envs), device)
    model = Network(prob_uniform=prob_uniform)
    model.to(device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    global global_step
    save_freq = parameters.get('save_freq', 100)

    for episode in trange(1, n_episodes + 1):

        # sample
        model.eval()
        with torch.no_grad():
            obs = env.reset()

            obss = []
            envs_done = torch.zeros(n_envs, dtype=bool).to(device=device)

            step_masks = []
            rewards = []
            dones = []
            values = []
            logprobs = []
            actions = []
            for t in range(T):
                global_step += (~envs_done).sum().item()
                obss.append(obs)
                value, a_dist = model(obs)
                values.append(value.squeeze(1))
                a = a_dist.sample()
                actions.append(a)
                obs, r, done, _ = env.step(a)
                rewards.append(r)
                step_masks.append(~envs_done)
                envs_done = envs_done | done
                dones.append(done)
                logprobs.append(a_dist.log_prob(a))

                for a_, prob_a in enumerate(a_dist.probs.mean(dim=0)):
                    logger.log_metric(f'p_action_{a_}', prob_a.item(), global_step)

                if envs_done.all():
                    break

            last_value = torch.zeros_like(r)  # not relevant
            advantages = [None] * len(rewards)
            returns = [None] * len(rewards)
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = ~done
                    next_return = last_value
                else:
                    nextnonterminal = ~dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages[t] = returns[t] - values[t]

            b_obs = torch.cat(obss)
            b_logprobs = torch.cat(logprobs)
            b_actions = torch.cat(actions)
            b_advantages = torch.cat(advantages)
            b_returns = torch.cat(returns)
            b_step_mask = torch.cat(step_masks)
            inds = torch.where(b_step_mask)[0].cpu().numpy()

        model.train()
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, len(inds) - minibatch_size + 1, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                mb_obs = b_obs[minibatch_ind]
                mb_actions = b_actions[minibatch_ind]
                mb_advantages = b_advantages[minibatch_ind]
                mb_logprobs = b_logprobs[minibatch_ind]

                new_values, dew_distr = model(mb_obs)
                newlogproba = dew_distr.log_prob(mb_actions)
                ratio = (newlogproba - mb_logprobs).exp()
                entropy_sum = dew_distr.entropy().sum()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = -entropy_sum / minibatch_size

                # Value loss
                v_loss = ((new_values - b_returns[minibatch_ind]) ** 2).mean()
                v_variance = b_returns[minibatch_ind].var()
                explained_var = 1 - (new_values - b_returns[minibatch_ind]).var() / v_variance

                # Total loss
                loss = pg_loss + ent_coef * entropy_loss + .5 * v_loss * vf_coef

                # Log losses
                logger.log_metric('loss_policy', pg_loss.item(), global_step)
                logger.log_metric('entropy', -entropy_loss.item(), global_step)
                logger.log_metric('loss_value', v_loss.item(), global_step)
                logger.log_metric('loss_total', loss.item(), global_step)
                logger.log_metric('variance_value', v_variance.item(), global_step)
                logger.log_metric('explained_var', explained_var.item(), global_step)
                logger.log_metric('step', global_step, global_step)
                opt.zero_grad()
                loss.mean().backward()

                opt.step()
        if episode % save_freq == 0 or episode == n_episodes:
            model_fname = f'model_ep_{episode}.pt'
            torch.save(model, model_fname)
            mlflow.log_artifact(model_fname)


if __name__ == '__main__':
    root = Path(__file__).absolute().parents[1]
    run_name = 'norm_input'
    user = 'linas'

    mmflow.run_remote('edm-gpu', memory=('4000Mi', '8000Mi'), gpu=True)
    run_edm_ppo(Path('edm_u0_e1e-2_fp1e-2.yml'), run_name=run_name, user=user)
