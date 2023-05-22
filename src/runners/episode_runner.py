from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
# TESTING


class EpisodeRunner:

    def __init__(self, args, logger):
        aa = 1
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
        print("EPISODE LIMIT:", self.episode_limit)
        self.t = 0

        self.t_env = 0

        self.fill_gamma()

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.ret = []

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

    def run(self, test_mode=False, log_results=False):
        if test_mode == False:
          self.reset()
        if test_mode == True:
          self.reset_test()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        curRew = [[],[],[],[]]
        cumRew = [0,0,0,0]
        episode_other = 0
        while not terminated:

            if test_mode:
              pre_transition_data = {
                # "state": [self.env.get_state_test()],
                # "avail_actions": [self.env.get_avail_actions_test()],
                "obs": [self.env.get_obs_test()]
              }
            else:
              pre_transition_data = {
                  # "state": [self.env.get_state()],
                  # "avail_actions": [self.env.get_avail_actions()],
                  "obs": [self.env.get_obs()]
              }

            # print("PRE_TRANSITION_DATA: ", pre_transition_data)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # print("ACTIONS: ", actions)

            if test_mode == True:
              reward, terminated, env_info, rewards = self.env.step_test(actions[0])
            else:
              reward, terminated, env_info, rewards = self.env.step(actions[0])
            
            for i in range(len(rewards)):
              rewards[i] *= -1
            # print("REWARDS2222:", rewards)
            episode_return += -reward
            # episode_return = 0.99*episode_return + reward


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
#             cumRew[0] += self.gamma[self.t]*rewards[0]
#             cumRew[1] += self.gamma[self.t]*rewards[1]
#             cumRew[2] += self.gamma[self.t]*rewards[2]
#             cumRew[3] += self.gamma[self.t]*rewards[3]

            post_transition_data = {
                "actions": actions,
                "reward": [(rewards[0],rewards[1],rewards[2],rewards[3])],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        episode_other = sum(cumRew)
        # if test_mode:
        #   print("EPISODE OTHER:")
        #   print(episode_other)
        self.ret += [cumRew]
        #torch.Size([1, 101, 4])
        totalRew = sum(cumRew) + 0.1
        for i in range(4):
          for t in range(self.t):
            curRew[i][t] += (40.0 / 3.0)*(totalRew - cumRew[i]) / self.t

        # print("data.tranisition_data")
        # print(self.batch.data.transition_data["reward"].size())
        # print(self.batch.data.transition_data["reward"][0])

        # print("CURREW:")
        # print(curRew)
        # set them all as curRew
        for t in range(self.t):
          for i in range(4):
            self.batch.data.transition_data["reward"][0][t][i] = curRew[i][t]

        # print("TRANS DATA")
        # print(self.batch.data.transition_data)
        # print("EPS DATA")
        # print(self.batch.data.episode_data)
        # print("BATCH.DATA")
        # print(type(self.batch.data.transition_data["reward"]))
        # print(self.batch.data.transition_data["reward"].size())

        # now the total reward for time t is sum of all agents at time t
        # for t in range(self.t):
        #   value = 0
        #   for i in range(4):
        #     value += curRew[i][t]
        #   self.batch.data.transition_data['reward'][0][t][0] = value

        if test_mode:
          last_data = {
            # "state": [self.env.get_state_test()],
            # "avail_actions": [self.env.get_avail_actions_test()],
            "obs": [self.env.get_obs_test()]
          }
        if test_mode == False:
          last_data = {
              # "state": [self.env.get_state()],
              # "avail_actions": [self.env.get_avail_actions()],
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

        # cur_returns.append(episode_return)
        cur_returns.append(episode_other)
        # if test_mode:
        #   print("EP RETURN: ", episode_return)

        if test_mode and log_results:
          p = self.ret
          # print(p)
          averages = [sum(column) / len(p) for column in zip(*p)]
          sum_of_averages = sum(averages)
          print("Sum of TEST:", sum_of_averages)
          print("CUMREWARD:", cumRew)
          print("EPS OTHER:", episode_other)
          self.logger.log_stat("new_return", sum_of_averages, self.t_env)
          self.ret = []

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            # print("SUM OF TEST: ", sum(cur_returns))
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

    def fill_gamma(self):
      self.gamma = [1.0,0.99,0.9801,0.970299,0.96059601,0.9509900498999999,0.941480149401,0.9320653479069899,0.9227446944279201,0.9135172474836408,0.9043820750088044,0.8953382542587164,0.8863848717161292,0.8775210229989678,0.8687458127689782,0.8600583546412884,0.8514577710948755,0.8429431933839268,0.8345137614500875,0.8261686238355866,0.8179069375972308,0.8097278682212584,0.8016305895390459,0.7936142836436554,0.7856781408072188,0.7778213593991467,0.7700431458051551,0.7623427143471035,0.7547192872036326,0.7471720943315961,0.7397003733882802,0.7323033696543975,0.7249803359578534,0.7177305325982749,0.7105532272722921,0.7034476949995692,0.6964132180495735,0.6894490858690777,0.682554595010387,0.6757290490602831,0.6689717585696803,0.6622820409839835,0.6556592205741436,0.6491026283684022,0.6426116020847181,0.6361854860638709,0.6298236312032323,0.6235253948912,0.617290140942288,0.611117239532865,0.6050060671375364,0.598956006466161,0.5929664464014994,0.5870367819374844,0.5811664141181095,0.5753547499769285,0.5696012024771592,0.5639051904523875,0.5582661385478637,0.5526834771623851,0.5471566423907612,0.5416850759668536,0.536268225207185,0.5309055429551132,0.525596487525562,0.5203405226503064,0.5151371174238033,0.5099857462495653,0.5048858887870696,0.4998370298991989,0.49483865960020695,0.4898902730042049,0.48499137027416284,0.4801414565714212,0.47534004200570695,0.4705866415856499,0.46588077516979337,0.46122196741809546,0.4566097477439145,0.45204365026647536,0.4475232137638106,0.4430479816261725,0.43861750180991077,0.43423132679181164,0.4298890135238935,0.4255901233886546,0.421334222154768,0.41712087993322033,0.41294967113388814,0.40882017442254925,0.4047319726783238,0.40068465295154054,0.3966778064220251,0.39271102835780486,0.3887839180742268,0.38489607889348454,0.38104711810454966,0.37723664692350417,0.37346428045426916,0.36972963764972644,0.3660323412732292,0.3623720178604969,]



