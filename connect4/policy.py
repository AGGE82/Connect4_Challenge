import numpy as np
from abc import ABC, abstractmethod
from connect_state import ConnectState
import math

class Policy(ABC):

    @abstractmethod
    def mount(self) -> None:
        pass

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass
    
class NodeMCTS():
    def __init__(self, state, action, parent=None):
        self.state = state
        self.visits = 0
        self.total_reward = 0
        self.parent = parent
        self.children = []
        self.action = action
        self.untried_actions = state.get_free_cols()
        
    def is_full_extended(self):
        if len(self.untried_actions) == 0:
            return True
        return False
    
    def q_estimate(self):
        q_value = self.total_reward/self.visits
        return q_value
    
    def best_child(self, c):
        best_child = None
        for child in self.children:
            best_child_UCB = 0
            
            if child.visits == 0:
                child_UCB = float('inf')
            else:
                child_UCB = child.q_estimate() + c * math.sqrt(math.log(self.parent.visits)/child.visits)
            
            if child_UCB > best_child_UCB:
                best_child_UCB = child_UCB
                best_child = child
        return best_child

class MCTS(Policy):
    def __init__(self):
        self.simulations = 1000
        self.c = 1.5
        self
        
    def mount(self):
        pass
    
    def act(self, s):
        node_root = NodeMCTS(s)
        best_action = None
        
        while self.simulations > 1000:
            self.selection(node_root)
            self.expansion()
            self.simulation()
            self.backpropagation()
            
        return best_action
            
    def selection(self, node):
        while node.is_full_extended() and not node.state.is_final():
            best_child = node.best_child(self.c)
        return best_child
    
    def expansion(self, node):
        untried_actions = node.state.get_free_cols()
        action = untried_actions[0]
        untried_actions = untried_actions.remove(action)
        
    
    def simulation(self):
        pass


    def backpropagation(self, reward, node):
        while node is not None:
            node.visitas += 1
            node.total_rewars += reward
            node = node.parent
    