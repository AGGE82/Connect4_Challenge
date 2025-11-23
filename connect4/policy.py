import numpy as np
import math
from abc import ABC, abstractmethod


from connect4.policy import Policy
from connect4.connect_state import ConnectState

class Policy(ABC):

    @abstractmethod
    def mount(self) -> None:
        pass

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass
