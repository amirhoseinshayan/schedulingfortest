"""muxGNN integration package.

The v2 Smoke-Train Bridge imports the adapter/model without eagerly importing the
older dynamic PPO environment.  PPO/environment symbols remain lazily available
for later stages, so this change only removes an unrelated import-time coupling.
"""

from .adapter import HeteroMuxAdapter
from .model import MuxActorCriticNetwork, MuxGNNEncoder

__all__ = [
    "HeteroMuxAdapter",
    "MuxGNNEncoder",
    "MuxActorCriticNetwork",
    "MuxGNNEventDrivenEnv",
    "MaskedMuxActorCriticPolicy",
]


def __getattr__(name: str):
    if name == "MuxGNNEventDrivenEnv":
        from .env import MuxGNNEventDrivenEnv

        return MuxGNNEventDrivenEnv
    if name == "MaskedMuxActorCriticPolicy":
        from .policy import MaskedMuxActorCriticPolicy

        return MaskedMuxActorCriticPolicy
    raise AttributeError(name)
