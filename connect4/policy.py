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
    
    def calc_reward(final_state):
        winner = final_state.get_winner()

        if winner == current_player: # AYUDA
            return 1
        elif winner == opponent:    #AYUDA
            return -1
        elif winner == 0 and final_state.is_final():
            return 0
        else:
            return 0
        
    def random_action(valid_actions):
        rng = np.random.default_rng()
        if len(valid_actions) == 0:
            return None
        random_index = int(rng.choice(valid_actions))
        action = valid_actions[random_index]
        return action
                
    def selection(self, node):
        while node.is_full_extended() and not node.state.is_final():
            best_child = node.best_child(self.c)
        return best_child
    
    def expansion(self, node):
        untried_actions = node.state.get_free_cols()
        action = untried_actions[0]
        untried_actions = untried_actions.remove(action)
        new_state = node.state.transition(action)
        child = NodeMCTS(new_state, action, node)
        node.childs.add(child)
        return child
    
    def simulation(self, state):
        current_state = state.copy()
        while not current_state.is_final():
           action = self.random_action(current_state.get_free_cols)
           current_state = current_state.transition(action)
        return self.calc_reward(current_state)

    def backpropagation(self):
        pass
    