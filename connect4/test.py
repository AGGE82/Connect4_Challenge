from policy import MCTS
import numpy as np
from connect_state import ConnectState
class MockState:
    """
    Dummy placeholder so test.py runs without errors.
    Replace with real ConnectState.
    """
    def __init__(self):
        # 6x7 Connect 4 empty board
        self.board = np.zeros((6, 7), dtype=int)
        self.player = 1

    def get_free_cols(self):
        # All columns are free (0–6)
        return list(range(7))

    def transition(self, action):
        # Return a new copy (fake transition)
        new_state = MockState()
        new_state.board = np.copy(self.board)
        new_state.board[0][action] = self.player
        new_state.player = -self.player
        return new_state

    def is_final(self):
        # Never final in dummy version
        return False

    def get_winner(self):
        # No winner
        return None

    def copy(self):
        new_state = MockState()
        new_state.board = np.copy(self.board)
        new_state.player = self.player
        return new_state


# ---------------------------------------------------------------------

def main():
    print("Starting MCTS test...")

    # Create MCTS policy
    policy = MCTS()
    policy.mount()

    # Create test state
    state = MockState()

    # Run action selection
    try:
        action = policy.act(state)
        print("MCTS selected action:", action)
    except Exception as e:
        print("An error occurred while running MCTS:")
        print(e)


if __name__ == "__main__":
    main()
