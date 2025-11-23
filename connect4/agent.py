import numpy as np
import math

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
        self.cached_ucb = None
        self._cached_parent_visits = None
        self._cached_child_visits = None

    def is_full_extended(self):
        return len(self.untried_actions) == 0

    def q_estimate(self):
        if self.visits == 0:
            return 0
        return self.total_reward / self.visits

    def best_child(self, c):
        parent_visits = self.visits if self.visits > 0 else 1

        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return np.random.choice(unvisited)

        best_child = None
        best_child_UCB = -float('inf')

        for child in self.children:
            if (child._cached_child_visits != child.visits or child._cached_parent_visits != parent_visits or child.cached_ucb is None):
                child.cached_ucb = (
                    child.q_estimate()
                    + c * math.sqrt(math.log(parent_visits) / child.visits)
                    )
                child._cached_child_visits = child.visits
                child._cached_parent_visits = parent_visits

            child_UCB = child.cached_ucb
            if child_UCB > best_child_UCB:
                best_child_UCB = child_UCB
                best_child = child

        return best_child

class MCTS(Policy):
    def __init__(self):
        self.simulations = 1000
        self.c = 1.5

    def mount(self):
        pass

    def detect_player(self, s: np.ndarray):
        p1 = np.sum(s == 1)
        p2 = np.sum(s == -1)
        return 1 if p1 == p2 else -1

    def act(self, s: np.ndarray) -> int:
        player = self.detect_player(s)
        state = ConnectState(s.copy(), player)

        if state.is_final():
            return 0

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
            node = node.parent

class BoardBuffer:
    def __init__(self, num=8, rows=6, cols=7):
        self.buffers = [np.zeros((rows, cols), dtype=int) for _ in range(num)]
        self.index = 0
        self.num = num

    def get(self):
        buffer = self.buffers[self.index]
        self.index = (self.index + 1) % self.num
        return buffer

def get_empty_board_buffer():
    return np.zeros((6, 7), dtype=int)

def transition_fast_board(board: np.ndarray, action: int, player: int, buffer: np.ndarray):
    np.copyto(buffer, board)
    col = buffer[:, action]
    for row in range(len(col)-1, -1, -1):
        if col[row] == 0:
            buffer[row, action] = player
            return True
    return False

def check_win_fast(board:np.ndarray, player:int) -> bool:
    rows, cols = board.shape

    for r in range(rows):
        row = board[r]
        for col in range(cols-3):
            if row[col] == player and row[col+1] == player and row[col+2] == player and row[col+3] ==player:
                return True
            
    for c in range(cols):
        col= board[:, c]
        for row in range(rows-3):
            if col[row] == player and col[row+1] == player and col[row+2] == player and col[row+3] == player:
                return True
            
    for row in range(rows-3):
        for col in range(cols-3):
            if (board[row, col] == player and board[row+1, col+1] == player and board[row+2, col+2] == player and board[row+3, col+3] == player):
                return True

    for row in range(3, rows):
        for col in range(cols-3):
            if (board[row, col] == player and board[row-1, col+1] == player and board[row-2, col+2] == player and board[row-3, col+3] == player):
                return True

    return False

def get_winner_board(board: np.ndarray) -> int:
    if check_win_fast(board, 1):
        return 1
    if check_win_fast(board, -1):
        return -1
    return 0