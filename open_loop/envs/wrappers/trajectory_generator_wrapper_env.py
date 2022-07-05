import gym


class TrajectoryGeneratorWrapperEnv(gym.Wrapper):

    """A wrapped GymEnv with a built-in trajectory generator."""

    def __init__(self, env, trajectory_generator):

        """Initialzes the wrapped env.

        Args:
          gym_env: An instance of LocomotionGymEnv.
          trajectory_generator: A trajectory_generator that can potentially modify
            the action and observation. Typticall generators includes the PMTG and
            openloop signals. Expected to have get_action and get_observation
            interfaces.

        Raises:
          ValueError if the controller does not implement get_action and
          get_observation.

        """
        super().__init__(env)

        if not hasattr(trajectory_generator, 'get_action') \
                or not hasattr(trajectory_generator, 'get_observation'):
            raise ValueError(
                'The controller does not have the necessary interface(s) implemented.'
            )

        self.trajectory_generator = trajectory_generator

        # The trajectory generator can subsume the action/observation space.
        if hasattr(trajectory_generator, 'observation_space'):
          self.observation_space = self.trajectory_generator.observation_space

        if hasattr(trajectory_generator, 'action_space'):
          self.action_space = self.trajectory_generator.action_space

        self._step_counter = 0

    def reset(self):
        if getattr(self.trajectory_generator, 'reset'):
            self.trajectory_generator.reset()
        observation = self.env.reset()
        self._step_counter = 0
        return self._modify_observation(observation)

    def step(self, action):
        """Steps the wrapped environment.

        Args:
          action: Numpy array. The input action from an NN agent.

        Returns:
          The tuple containing the modified observation, the reward, the epsiode end
          indicator.

        Raises:
          ValueError if input action is None.

        """
        if action is None:
          raise ValueError('Action cannot be None')

        traj_action = self.trajectory_generator.get_action(
            self.get_time_since_reset())

        new_action = traj_action
        original_observation, reward, done, info = self.env.step(new_action)
        self._step_counter += 1

        return self._modify_observation(original_observation), reward, done, info

    def get_time_since_reset(self):
        return self.env.dt * self._step_counter

    def _modify_observation(self, observation):
        return self.trajectory_generator.get_observation(observation)
