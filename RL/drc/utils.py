class JumanjiEnvAdapter:
    def __init__(self, env):
        self._env = env
        self.action_spec = env.action_spec

    def reset(self, keys):
        env_state, timestep = self._env.reset(keys)
        obs = timestep.observation.grid
        return env_state, obs

    def step(self, env_state, action):
        env_state, timestep = self._env.step(env_state, action)
        obs = timestep.observation.grid
        reward = timestep.reward
        done = timestep.last()
        return env_state, obs, reward, done


def setup_env(args):
    import jumanji
    from jumanji import wrappers

    if args.env_id != "Sokoban-v0":
        raise ValueError(f"Only Sokoban-v0 is supported, got {args.env_id}")
    env = jumanji.make(args.env_id)

    env = wrappers.VmapAutoResetWrapper(env)
    return JumanjiEnvAdapter(env)
