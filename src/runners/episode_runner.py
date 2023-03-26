from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        print("ARGS: ")
        print(self.args.env)
        print(self.args.env_args)
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # print("HEREEEEE: ")
        # print(self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
    
    def reset_test(self):
        self.batch = self.new_batch()
        self.env.reset_test()
        self.t = 0

    def run(self, test_mode=False):
        if test_mode == False:
          self.reset()
        if test_mode == True:
          self.reset_test()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        curRew = [[],[],[],[]]
        cumRew = [0,0,0,0]
        while not terminated:

            if test_mode:
              pre_transition_data = {
                #"state": [self.env.get_state_test()],
                "avail_actions": [self.env.get_avail_actions_test()],
                "obs": [self.env.get_obs_test()]
              }
            else:
              pre_transition_data = {
                  #"state": [self.env.get_state()],
                  "avail_actions": [self.env.get_avail_actions()],
                  "obs": [self.env.get_obs()]
              }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            if test_mode == True:
              reward, terminated, env_info, rewards = self.env.step_test(actions[0])
            else:
              reward, terminated, env_info, rewards = self.env.step(actions[0])
            
            episode_return += reward

            # if rewards at end is a list
            # accumulate cumReward with args.gamma
            curRew[0].append(rewards[0])
            curRew[1].append(rewards[1])
            curRew[2].append(rewards[2])
            curRew[3].append(rewards[3])
            cumRew[0] = self.args.gamma*cumRew[0] + rewards[0]
            cumRew[1] = self.args.gamma*cumRew[1] + rewards[1]
            cumRew[2] = self.args.gamma*cumRew[2] + rewards[2]
            cumRew[3] = self.args.gamma*cumRew[3] + rewards[3]

          
            post_transition_data = {
                "actions": actions,
                "reward": [(rewards[0],rewards[1],rewards[2],rewards[3])],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        #torch.Size([1, 101, 4])
        totalRew = sum(cumRew)
        for i in range(4):
          for t in range(self.t):
            curRew[i][t] += (20.0 / 3.0)*(totalRew - cumRew[i]) / self.t

        # print("data.tranisition_data")
        # print(self.batch.data.transition_data["reward"].size())
        # print(self.batch.data.transition_data["reward"][0])

        # set them all as curRew
        for t in range(self.t):
          for i in range(4):
            self.batch.data.transition_data["reward"][0][t][i] = curRew[i][t]

        # now the total reward for time t is sum of all agents at time t
        # for t in range(self.t):
        #   value = 0
        #   for i in range(4):
        #     value += curRew[i][t]
        #   self.batch.data.transition_data['reward'][0][t][0] = value

        if test_mode:
          last_data = {
            #"state": [self.env.get_state_test()],
            "avail_actions": [self.env.get_avail_actions_test()],
            "obs": [self.env.get_obs_test()]
          }
        if test_mode == False:
          last_data = {
              #"state": [self.env.get_state()],
              "avail_actions": [self.env.get_avail_actions()],
              "obs": [self.env.get_obs()]
          }
        self.batch.update(last_data, ts=self.t)
        # print("AFTER")

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        # print("ACTIONS")

        #print(self.batch.data.transition_data['reward'])

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        # if test_mode:
        #   print("EP RETURN: ", episode_return)


        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            print("SUM OF TEST: ", sum(cur_returns))
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        # print("LEFT")
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
