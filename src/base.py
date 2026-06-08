# =============================================================================
# DO NOT MODIFY — provided infrastructure
# =============================================================================
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    def __init__(self, obs_dim: int, n_actions: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    @abstractmethod
    def train(self, env, n_steps: int = 200_000) -> None: ...

    @abstractmethod
    def act(self, obs: np.ndarray) -> int: ...
