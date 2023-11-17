import multiprocessing
import multiprocessing.connection
import concurrent.futures
import threading
import time

from random import randint

from neroRRL.environments.wrapper import wrap_environment

def worker_process(remote: multiprocessing.connection.Connection, env_seed, env_config, worker_id: int, record_video = False):
    """Initializes the environment and executes its interface.

    Arguments:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_seed {int} -- Sampled seed for the environment worker to use
        env_config {dict} -- The configuration data of the desired environment
        worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
    """
    import numpy as np
    np.random.seed(env_seed)
    import random
    random.seed(env_seed)
    random.SystemRandom().seed(env_seed)

    # Initialize and wrap the environment
    try:
        if "buffered_env" in env_config:
            env = BufferedEnv(env_config, worker_id, record_video, env_config["buffered_env"]["buffer_size"], env_config["buffered_env"]["n_processes"])
        else:
            env = wrap_environment(env_config, worker_id, record_trajectory = record_video)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset(data))
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            elif cmd == "video":
                remote.send(env.get_episode_trajectory)
            elif cmd == "image":
                remote.send(env.image)
            else:
                raise NotImplementedError
        except Exception as e:
            raise WorkerException(e)

class Worker:
    """A worker that runs one thread and controls its own environment instance."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_config, worker_id: int, record_video = False):
        """
        Arguments:
            env_config {dict} -- The configuration data of the desired environment
            worker_id {int} -- worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
        """
        env_seed = randint(0, 2 ** 32 - 1)
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_seed, env_config, worker_id, record_video))
        self.process.start()

    def close(self):
        self.child.send(("close", None))
        self.child.recv()
        self.process.join()
        self.process.terminate()


import tblib.pickling_support
tblib.pickling_support.install()
import sys
class WorkerException(Exception):
    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()
        super(WorkerException, self).__init__(str(ee))

    def re_raise(self):
        raise (self.ee, None, self.tb)

class BufferedEnv:
    """This class implements a buffered environment for parallel simulation and interaction.
    
    It keeps a buffer of environments that are ready to interact with,
    and in parallel, it resets the environments that were interacted with in a separate process pool.
    This allows to hide the reset time of the environments, which can be significant in some cases.
    """
    def __init__(self, env_config, worker_id, record_video, buffer_size, n_processes):
        """
        Initializes the BufferedEnv class.

        Arguments:
            env_config {dict} -- Configuration for the environment.
            worker_id {int} -- ID of the worker.
            record_video {bool} -- Boolean flag to decide if trajectories should be recorded.
            buffer_size {int} -- The size of the buffer to store ready environments.
            n_processes {int} -- The number of processes in the process pool.
        """
        self.running = True
        self.buffer_size = buffer_size
        self.env_config = env_config
        self.worker_id = worker_id
        self.record_video = record_video
        self.n_processes = n_processes
        # Create a list of ready environments
        self.ready_envs = [self.init_env(env_config) for _ in range(buffer_size)]
        self.to_reset_envs = []
        # Create a process pool executor for resetting environments
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_processes)
        # Lock for thread-safe operations on the ready_envs and to_reset_envs lists
        self.lock = threading.Lock()
        # Pop the first environment and observation from the ready environments
        self.env, _ = self.ready_envs.pop(0)
        
        # Start the fill_buffer method in a separate thread
        self.buffer_thread = threading.Thread(target=self.fill_buffer, daemon=True)
        self.buffer_thread.start()

    def init_env(self, env_config, worker_id=0, record_video=False):
        """
        Initialize an environment with the given configuration.

        Arguments:
            env_config {dict} -- Configuration for the environment.
            worker_id {int} -- ID of the worker.
            record_video {bool} -- Boolean flag to decide if trajectories should be recorded.

        Returns:
            {tuple} -- A tuple containing the environment and initial observation.
        """
        env = wrap_environment(env_config, worker_id, record_trajectory=record_video)
        return env, env.reset()

    @staticmethod
    def reset_env(env, reset_params):
        """Resets the given environment.

        Arguments:
            env {Environment} -- The environment to reset.
        
        Returns:
            {tuple} -- A tuple containing the environment and initial observation after reset.
        """
        return env, env.reset(reset_params)

    def fill_buffer(self):
        """
        Fill the buffer with ready environments by resetting environments in the to_reset_envs list.

        This method runs indefinitely and is meant to be run in a separate thread.
        """
        while self.running:
            with self.lock:
                # Reset all environments in the to_reset_envs list
                while self.to_reset_envs:
                    env, reset_params = self.to_reset_envs.pop(0)
                    future = self.executor.submit(BufferedEnv.reset_env, env, reset_params)
                    self.ready_envs.append(future.result())
            # Prevent high CPU usage
            time.sleep(0.01)

    def step(self, *args):
        """
        Takes a step in the environment using the given action.

        Arguments:
            *args -- Arguments to pass to the environment's step method.

        Returns:
            {tuple} -- A tuple of (observation, reward, done, info).
        """
        return self.env.step(*args)

    def reset(self, reset_params):
        """
        Resets the current environment.

        It first waits until there is at least one ready environment,
        then it moves the current environment to the to_reset_envs list,
        and then it pops a new environment from the ready_envs list.

        Arguments:
            {reset_params} -- Reset parameters, if any, to be passed to the environment's reset method.

        Returns:
            {np.array} -- The initial observation from the new environment.
        """
        # Wait until there is at least one ready environment
        while not self.ready_envs:
            time.sleep(0.01)  # prevent high CPU usage
        with self.lock:
            # Add the current environment to the to_reset_envs list
            self.to_reset_envs.append((self.env, reset_params))
            # Pop a new environment from the ready_envs list
            self.env, obs = self.ready_envs.pop(0)
        return obs
    
    @property
    def image(self):
        """ Returns the image of the current step."""
        return self.env.image
    
    def close(self):
        """
        Close the BufferedEnv object.

        This method stops the fill_buffer thread and shuts down the executor. Also it cleans up the resources used by the environments.
        """
        # Signal the fill_buffer thread to stop
        self.running = False
        # Wait for the fill_buffer thread to finish
        self.buffer_thread.join()
        # Shutdown the executor
        self.executor.shutdown()
        # Close the environments
        self.env.close()
        for env, _ in self.ready_envs:
            env.close()
        for env in self.to_reset_envs:
            env.close()