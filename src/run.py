import datetime
import os
import pprint
import time
import threading
from numpy import argsort
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
import logging
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import random
import pickle
import copy

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)


    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        # "nextobs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        # "avail_actions": {
        #     "vshape": (env_info["n_actions"],),
        #     "group": "agents",
        #     "dtype": th.int,
        # },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    if args.load_buffer != "":
      
      try:
        with open('buffer.pkl', 'rb') as file:
          buffer = pickle.load(file)
        episode = buffer.episodes_in_buffer
      except (pickle.UnpicklingError, FileNotFoundError) as e:
        print("Error occurred while loading the pickled object:", e)
      
    else:
      buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
      buffer2 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
      episode = 0

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            print("full_name: ", full_name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        # if args.evaluate or args.save_replay:
        #     runner.log_train_stats_t = runner.t_env
        #     evaluate_sequential(args, runner)
        #     logger.log_stat("episode", runner.t_env, runner.t_env)
        #     logger.print_recent_stats()
        #     logger.console_logger.info("Finished Evaluation")
        #     return

    # start training
    # episode = 0 # Set about for if loaded buffer
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = runner.t_env

    start_time = time.time()
    last_time = start_time 

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # torch.backends.cudnn.benchmark = True
    ep = 0
    while runner.t_env <= args.t_max:

        # Play an entire episode and get batch of experiences
        episode_batch = runner.run(buffer, args, learner, episode, test_mode=False)
        
        # Insert batch into buffer
        buffer.insert_episode_batch(episode_batch)
        if ep > 64:
            x = 0
            # buffer2.insert_episode_batch(episode_batch)
        if ep == -128:
            buffer = copy.deepcopy(buffer2)
            ep = 0
        ep += 1

        
        # Train on large number of experiences
        # Before: Would sample n episodes and train on them
        # Now: Creates a temp batch of random experiences
        # Maybe try if > 500 and episode % 20 == 0, then use very large batch
        if episode > 64 and buffer.can_sample(64): #args.batch_size):
          
          # TODO: make this able to be set from here?
          # Create new (empty) batch of some size
          # r = runner.get_sample()
          # print("SAMPLE:")
          # print(r)
          # print("TO DEVICE")
        # def to(self, device):
        # for k, v in self.data.transition_data.items():
        #     self.data.transition_data[k] = v.to(device)
        # for k, v in self.data.episode_data.items():
        #     self.data.episode_data[k] = v.to(device)
        # self.device = device

          # For 10 times,
          #   Get 32 episodes from buffer
          #   Train on a random continuous 5 steps from those episodes
          # Each iteration does 32*5 and there are 10 of them = 1600
          # If after each action we did an episode as a batch
          # that would be 40*40 = 1600
          # So this is equal to that but mixes it up more
         
          for _ in range(7): #10):
            new_batch = None #runner.new_batch64()
            episode_sample = buffer.sample(args.batch_size, args, learner, runner.t_env, new_batch)
         
            if episode_sample != None:

              # Truncate batch to only filled timesteps
              max_ep_t = episode_sample.max_t_filled()
              episode_sample = episode_sample[:, :max_ep_t]

              if episode_sample.device != args.device:
                episode_sample.to(args.device)
              

              learner.train(episode_sample, runner.t_env, episode)
        

        # Execute test runs once in a while
        # Hardcoded to 50 for beer game
        n_test_runs = 50 #max(1, args.test_nepisode // runner.batch_size)
      
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env

            # On last test episode, let know to log result of test runs
            for x in range(n_test_runs):
              if x == n_test_runs - 1:
                runner.run(buffer, args, learner, episode, test_mode=True,log_results=True)
              else:
                runner.run(buffer, args, learner, episode, test_mode=True, log_results=False)

        # if args.save_model and (
        #     runner.t_env - model_save_time >= args.save_model_interval
        #     or model_save_time == 0
        # ):
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
    
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            with open('buffer.pkl', 'wb') as file:
              pickle.dump(buffer, file)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
