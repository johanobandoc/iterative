from dataclasses import dataclass, replace as dataclass_replace

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DiscreteActionSpec:
    num_values: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JumanjiNoopState:
    inner_state: object
    timestep: object

    @property
    def key(self):
        return self.inner_state.key

    def replace(self, **kwargs):
        inner_state = kwargs.pop("inner_state", self.inner_state)
        timestep = kwargs.pop("timestep", self.timestep)
        if "key" in kwargs:
            inner_state = dataclass_replace(inner_state, key=kwargs.pop("key"))
        if kwargs:
            raise TypeError(f"Unexpected replace fields: {', '.join(sorted(kwargs))}")
        return JumanjiNoopState(inner_state=inner_state, timestep=timestep)

    def tree_flatten(self):
        return (self.inner_state, self.timestep), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        inner_state, timestep = children
        return cls(inner_state=inner_state, timestep=timestep)


class JumanjiNoopActionWrapper:
    """Wrap a single Jumanji env to add a no-op action index."""

    def __init__(self, env, noop_reward: float = -0.01):
        self._env = env
        action_spec = env.action_spec
        self._base_num_values = int(action_spec.num_values)
        self._noop_reward = float(noop_reward)
        self.action_spec = DiscreteActionSpec(num_values=self._base_num_values + 1)
        self.observation_spec = env.observation_spec
        self.reward_spec = env.reward_spec
        self.discount_spec = env.discount_spec

    def reset(self, key):
        env_state, timestep = self._env.reset(key)
        return JumanjiNoopState(env_state, timestep), timestep

    def step(self, env_state, action):
        inner_state = env_state.inner_state
        last_timestep = env_state.timestep
        action = jnp.asarray(action)
        noop_index = jnp.asarray(self._base_num_values, dtype=action.dtype)
        is_noop = action == noop_index

        def do_noop(_):
            reward = jnp.full_like(last_timestep.reward, self._noop_reward)
            timestep = dataclass_replace(last_timestep, reward=reward)
            return JumanjiNoopState(inner_state, timestep), timestep

        def do_step(_):
            inner_state_next, timestep_next = self._env.step(inner_state, action)
            return JumanjiNoopState(inner_state_next, timestep_next), timestep_next

        return jax.lax.cond(is_noop, do_noop, do_step, operand=None)


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
        reward = jnp.where(reward < 0.0, jnp.asarray(-0.01, dtype=reward.dtype), reward)
        done = timestep.last()
        return env_state, obs, reward, done


def _make_jumanji_env(args):
    import jumanji
    from jumanji import wrappers

    env = jumanji.make(args.env_id)
    if args.env_id in {"lbf", "connector", "search_and_rescue"}:
        # Convert a multi-agent environment to a single-agent environment
        env = wrappers.MultiToSingleWrapper(env)

    if args.use_noop_action:
        env = JumanjiNoopActionWrapper(env)
    env = wrappers.VmapAutoResetWrapper(env)
    return JumanjiEnvAdapter(env)


class CraftaxEnvAdapter:
    def __init__(
        self,
        env,
        env_params,
    ):
        self._env = env
        self._env_params = env_params
        self.action_spec = DiscreteActionSpec(num_values=env.action_space(env_params).n)

    def _reset_one(self, key):
        obs, state = self._env.reset(key, self._env_params)
        return state, obs

    def reset(self, keys):
        split_keys = jax.vmap(jax.random.split)(keys)
        reset_keys = split_keys[:, 0]
        step_keys = split_keys[:, 1]
        state, obs = jax.vmap(self._reset_one)(reset_keys)
        env_state = (state, step_keys)
        return env_state, obs

    def step(self, env_state, action):
        state, key = env_state

        def step_one(state, action, key):
            key, step_key = jax.random.split(key)
            obs, state, reward, done, info = self._env.step(step_key, state, action, self._env_params)
            return (state, key), obs, reward, done

        (state, key), obs, reward, done = jax.vmap(step_one)(state, action, key)
        return (state, key), obs, reward, done


class CraftaxOptimisticResetEnvAdapter:
    def __init__(
        self,
        env,
        env_params,
        num_envs: int,
        reset_ratio: int,
    ):
        self._env = env
        self._env_params = env_params
        self._num_envs = num_envs
        self._reset_ratio = reset_ratio
        assert num_envs % reset_ratio == 0, "craftax_optimistic_reset_ratio must divide num_envs"
        self._num_resets = num_envs // reset_ratio
        self.action_spec = DiscreteActionSpec(num_values=env.action_space(env_params).n)

    def _reset_one(self, key):
        obs, state = self._env.reset(key, self._env_params)
        return state, obs

    def reset(self, keys):
        split_keys = jax.vmap(jax.random.split)(keys)
        reset_keys = split_keys[:, 0]
        step_keys = split_keys[:, 1]
        state, obs = jax.vmap(self._reset_one)(reset_keys)
        env_state = (state, step_keys)
        return env_state, obs

    def step(self, env_state, action):
        state, key = env_state

        def step_one(state, action, key):
            key, step_key, reset_key, choose_key = jax.random.split(key, 4)
            obs, state, reward, done, info = self._env.step(step_key, state, action, self._env_params)
            return (state, key, reset_key, choose_key), obs, reward, done

        (state_st, key, reset_key, choose_key), obs_st, reward, done = jax.vmap(step_one)(state, action, key)
        state_re, obs_re = jax.vmap(self._reset_one)(reset_key[: self._num_resets])

        reset_indexes = jnp.arange(self._num_resets).repeat(self._reset_ratio)
        done_probs = done.astype(jnp.float32)
        uniform_probs = jnp.full((self._num_envs,), 1.0 / self._num_envs, dtype=jnp.float32)
        done_count = done_probs.sum()
        done_probs = jnp.where(done_count > 0.0, done_probs / jnp.maximum(done_count, 1.0), uniform_probs)
        being_reset = jax.random.choice(
            choose_key[0],
            jnp.arange(self._num_envs),
            shape=(self._num_resets,),
            p=done_probs,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self._num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_re)

        def auto_reset(done_i, state_re_i, state_st_i, obs_re_i, obs_st_i):
            state_i = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done_i, x, y),
                state_re_i,
                state_st_i,
            )
            obs_i = jax.lax.select(done_i, obs_re_i, obs_st_i)
            return state_i, obs_i

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)
        return (state, key), obs, reward, done


def _make_craftax_env(args) -> CraftaxEnvAdapter:
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name(args.env_id, auto_reset=not args.craftax_optimistic_resets)
    env_params = env.default_params

    if args.craftax_optimistic_resets:
        env = CraftaxOptimisticResetEnvAdapter(
            env,
            env_params,
            num_envs=args.num_envs,
            reset_ratio=min(args.craftax_optimistic_reset_ratio, args.num_envs),
        )
    else:
        env = CraftaxEnvAdapter(env, env_params)
    return env


def setup_env(args):
    if "craftax" in args.env_id.lower():
        return _make_craftax_env(args)
    return _make_jumanji_env(args)
