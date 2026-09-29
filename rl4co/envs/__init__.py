"""CAT-focused RL4CO environment registry."""

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.routing.evrptw.env import EVRPTWEnv

ENV_REGISTRY = {"evrptw": EVRPTWEnv}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Create one of the environments included in this snapshot."""
    env_cls = ENV_REGISTRY.get(env_name)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name!r}. Available environments: {tuple(ENV_REGISTRY)}"
        )
    return env_cls(*args, **kwargs)
