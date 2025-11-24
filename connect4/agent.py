import json
import numpy as np
import math
import time
import os

from policy import Policy
from connect_state import ConnectState


class NodeMCTS():
    def __init__(self, state: ConnectState, action, parent=None):
        if state is None:
            raise ValueError("NodeMCTS requires a ConnectState 'state' (no None).")
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
        self.q_prior = 0

    def is_full_extended(self):
        return len(self.untried_actions) == 0

    def q_estimate(self):
        if self.visits == 0:
            return self.q_prior
        return self.total_reward / self.visits

    def best_child(self, c):
        parent_visits = self.visits if self.visits > 0 else 1

        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return np.random.choice(unvisited)

        best_child = None
        best_child_UCB = -float('inf')

        for child in self.children:
            if (child._cached_child_visits != child.visits or
                child._cached_parent_visits != parent_visits or
                child.cached_ucb is None):

                denom = child.visits if child.visits > 0 else 1.0
                ucb = child.q_estimate() + c * math.sqrt(math.log(parent_visits) / denom)

                child.cached_ucb = ucb
                child._cached_child_visits = child.visits
                child._cached_parent_visits = parent_visits

            child_UCB = child.cached_ucb
            if child_UCB > best_child_UCB:
                best_child_UCB = child_UCB
                best_child = child

        return best_child

class Bradwurst(Policy):
    qvalues = {}

    def __init__(self):
        self.simulations = 0
        self.c = 1.5
        self.buffer_pool = self.BoardBufferPool(num_buffers=16, rows=6, cols=7)
        self.max_rollout_depth = 6
        self.prune_min_visits = 1
        self.time_limit = 8
        self.root = None
        self.root_player = None
        self.qfile = os.path.join(os.path.dirname(__file__), "qvalues.json")

        if os.path.exists(self.qfile):
            try:
                with open(self.qfile, "r") as f:
                    self.qvalues = json.load(f)
            except Exception:
                self.qvalues = {}
        else:
            self.qvalues = {}

    def mount(self, timeout=None):
        self.time_limit = timeout if timeout is not None else self.time_limit

        if not os.path.exists(self.qfile):
            with open(self.qfile, "w") as f:
                json.dump({}, f)

    def detect_player(self, s: np.ndarray):
        p1 = np.sum(s == 1)
        p2 = np.sum(s == -1)
        return 1 if p1 == p2 else -1
    
    def update_root_after_move(self, prev_root: NodeMCTS, action: int, new_state: ConnectState):
        if prev_root is None:
            if new_state is None:
                raise ValueError("update_root_after_move called with prev_root=None y new_state=None")
            return NodeMCTS(new_state, None)

        if new_state is None and action is not None:
            try:
                candidate = prev_root.state.transition(action)
                new_state = candidate
            except Exception:
                new_state = None

        for child in prev_root.children:
            if child.action == action:
                child.parent = None
                if new_state is not None:
                    child.state = new_state
                    child.untried_actions = new_state.get_free_cols()[:]
                return child

        if new_state is None:
            if action is not None:
                try:
                    new_state = prev_root.state.transition(action)
                except Exception:
                    new_state = prev_root.state
            else:
                new_state = prev_root.state

        return NodeMCTS(new_state, None)
    
    def prune_tree(self, node: NodeMCTS):
        if not node.children:
            return
        node.children = [child for child in node.children if child.visits >= self.prune_min_visits]

    def act(self, s: np.ndarray) -> int:
        player = self.detect_player(s)
        state = ConnectState(s.copy(), player)

        if state.is_final():
            return 0

        valid_actions = state.get_free_cols()
        if not valid_actions:
            return 0

        for col in valid_actions:
            if state.transition(col).get_winner() == player:
                return col

        for col in valid_actions:
            if state.transition(col).get_winner() == -player:
                return col

        if self.root is None:
            node_root = NodeMCTS(state, None)
        else:
            node_root = self.update_root_after_move(self.root, None, state)
        self.root_player = player

        self.simulations = 0
        MAX_SIMULATIONS = 20
        
        for _ in range(MAX_SIMULATIONS):
            node = self.selection(node_root)
            node = self.expansion(node)
            reward = self.simulation(node.state)
            self.backpropagation(reward, node)
            self.simulations += 1

        if not node_root.children:
            return int(np.random.choice(valid_actions))

        best_child = max(node_root.children, key=lambda c: c.visits)

        new_state = node_root.state.transition(best_child.action)
        self.root = self.update_root_after_move(node_root, best_child.action, new_state)

        return best_child.action
    
    def calc_reward(self, winner, player):
        if winner == player:
            return 1
        elif winner == -player:
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
        if node.state.is_final():
            return node

        valid_actions = node.state.get_free_cols()
        if not valid_actions:
            node.untried_actions = []
            return node

        node.untried_actions = [a for a in node.untried_actions if a in valid_actions]

        if not node.untried_actions:
            return node

        action = int(np.random.choice(node.untried_actions))
        try:
            node.untried_actions.remove(action)
        except ValueError:
            pass

        new_state = node.state.transition(action)
        child = NodeMCTS(new_state, action, node)

        stateKey = str(node.state.board.tolist())
        if stateKey in self.qvalues and str(action) in self.qvalues[stateKey]:
            child.q_prior = self.qvalues[stateKey][str(action)]
        else:
            child.q_prior = 0

        node.children.append(child)
        return child

    def simulation(self, state: ConnectState):
        board = state.board
        current_player = state.player
        pool = self.buffer_pool
        buffer = pool.get()
        np.copyto(buffer, board)

        depth = 0
        while depth < self.max_rollout_depth:
            winner = get_winner_board(buffer)
            if winner != 0:
                return self.calc_reward(winner, self.root_player)
            free_cols = [col for col in range(buffer.shape[1]) if buffer[0, col] == 0]
            if not free_cols:
                return 0

            for action in free_cols:
                tmp = pool.get()
                if transition_fast_board(buffer, action, current_player, tmp):
                    if check_win_fast(tmp, current_player):
                        if current_player == self.root_player:
                            return 1
                        else:
                            return -1

            action = np.random.default_rng().choice(free_cols)
            transition_fast_board(buffer, action, current_player, buffer)
            current_player = -current_player
            depth += 1

        score = self.heuristic_board_score(buffer, self.root_player)
        if score > 0:
            return 0.5
        elif score < 0:
            return -0.5
        else:
            return 0
        
    def heuristic_board_score(self, board: np.ndarray, player: int) -> float:
        my = 0
        opp = 0
        rows, cols = board.shape

        def count_line(line):
            s_my = 0
            s_opp = 0
            for i in range(len(line)-3):
                window = line[i:i+4]
                if np.all(window == player):
                    s_my += 10
                else:
                    cnt_my = int(np.sum(window == player))
                    cnt_opp = int(np.sum(window == -player))
                    if cnt_my == 3 and cnt_opp ==0:
                        s_my +=4
                    elif cnt_my == 2 and cnt_opp == 0:
                        s_my += 1
                    if cnt_opp == 3 and cnt_my == 0:
                        s_opp += 4
                    elif cnt_opp == 2 and cnt_my == 0:
                        s_opp += 1
            return s_my, s_opp
        
        for row in range(rows):
            a, b = count_line(board[row])
            my += a
            opp += b

        for col in range(cols):
            a, b = count_line(board[:, col])
            my += a
            opp += b

        for row in range(rows - 3):
            for col in range(cols - 3):
                window =  np.array([board[row+i, col+i] for i in range(4)])
                a, b = count_line(window)
                my += a
                opp += b
        
        for row in range(3, rows):
            for col in range(cols - 3):
                window =  np.array([board[row-i, col+i] for i in range(4)])
                a, b = count_line(window)
                my += a
                opp += b

        return my - opp

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
