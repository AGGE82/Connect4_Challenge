import json
import numpy as np
import math
import os

from policy import Policy
from connect_state import ConnectState


class NodeMCTS():
    def __init__(self, state: ConnectState, action, parent=None):
        self.state = state
        self.visits = 0
        self.total_reward = 0
        self.parent = parent
        self.children = []
        self.action = action
        self.untried_actions = state.get_free_cols()[:]

    def is_full_extended(self):
        return len(self.untried_actions) == 0

    def q_estimate(self):
        if self.visits == 0:
            return 0
        return self.total_reward / self.visits

    def best_child(self, c):
        parent_visits = max(self.visits, 1)

        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return np.random.choice(unvisited)

        best_child = None
        best_child_UCB = -float('inf')

        for child in self.children:
            child_UCB = (
                child.q_estimate()
                + c * math.sqrt(math.log(parent_visits) / child.visits)
            )
            if child_UCB > best_child_UCB:
                best_child_UCB = child_UCB
                best_child = child

        return best_child


class MCTS(Policy):
    qvalues = {}

    def __init__(self):
        self.simulations = 100
        self.c = 1.5
        self.qfile = os.path.join(os.path.dirname(__file__), "qvalues.json")

        if os.path.exists(self.qfile):
                with open(self.qfile, "r") as f:
                    self.qvalues = json.load(f)
        else:
            self.qvalues = {}


    def mount(self):
        if not os.path.exists(self.qfile):
            with open(self.qfile, "w") as f:
                json.dump({}, f)

    def detect_player(self, s: np.ndarray):
        p1 = np.sum(s == 1)
        p2 = np.sum(s == -1)
        return 1 if p1 == p2 else -1

    def act(self, s: np.ndarray) -> int:
        player = self.detect_player(s)
        state = ConnectState(s.copy(), player)
        stateKey = str(state.board.tolist())

        if state.is_final():
            return 0

        if stateKey in self.qvalues:
            maxQ = max(self.qvalues[stateKey], key=self.qvalues[stateKey].get)
            return int(maxQ)


        valid_actions = state.get_free_cols()
        if not valid_actions:
            return 0

        for col in valid_actions:
            next_s = state.transition(col)
            if next_s.get_winner() == player:
                return col

        opp = -player
        for col in valid_actions:
            opp_state = ConnectState(state.board.copy(), opp).transition(col)
            if opp_state.get_winner() == opp:
                return col

        self.root_player = player
        node_root = NodeMCTS(state, None)

        for _ in range(self.simulations):
            node = self.selection(node_root)

            if not node.state.is_final():
                node = self.expansion(node)

            reward = self.simulation(node.state)
            self.backpropagation(reward, node)

        if not node_root.children:
            return np.random.choice(valid_actions)

        best_child = max(node_root.children, key=lambda c: c.visits)
        return best_child.action

    def calc_reward(self, final_state: ConnectState):
        winner = final_state.get_winner()
        if winner == self.root_player:
            return 1
        elif winner == -self.root_player:
            return -1
        return 0

    def random_action(self, valid_actions):
        if not valid_actions:
            return None
        return int(np.random.default_rng().choice(valid_actions))

    def selection(self, node):
        while node.is_full_extended() and not node.state.is_final():

            if not node.children:
                break

            node = node.best_child(self.c)

        return node

    def expansion(self, node):
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()

        if action not in node.state.get_free_cols():
            return node

        new_state = node.state.transition(action)
        child = NodeMCTS(new_state, action, node)
        node.children.append(child)

        return child

    def simulation(self, state: ConnectState):
        current_state = state

        while not current_state.is_final():
            action = self.random_action(current_state.get_free_cols())
            current_state = current_state.transition(action)

        return self.calc_reward(current_state)

    def backpropagation(self, reward, node):
        while node is not None:
            node.visits += 1
            node.total_reward += reward

            #Para guardar los q values
            stateKey = str(node.state.board.tolist())

            if stateKey not in self.qvalues:
                self.qvalues[stateKey] = {}

            if node.action is not None:
                q = node.total_reward / node.visits
                self.qvalues[stateKey][str(node.action)] = q

            node = node.parent

        with open(self.qfile, "w") as f:
            json.dump(self.qvalues, f, indent=2)    
