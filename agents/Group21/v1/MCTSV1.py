import copy
import math
import time
from random import choice

from agents.Group21.v1.DisjointSetBoardV1 import DisjointSetBoardV1
from agents.Group21.v1.MCTSNodeV1 import MCTSNodeV1
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSV1:

    # Hyperparameters
    EXPLORATION_WEIGHT = 1.0
    RAVE_K = 100

    SWAP_MOVE = -12 # (-1, -1) maps to -12

    def __init__(self, colour: Colour):
        self.colour = colour
        self._root: MCTSNodeV1 | None = None

    def run(self, time_limit: float = 0.5, iterations: int = 2000) -> Move:
        assert self._root is not None, "Call update(board, opp_move) before run() to set root."

        end_time = time.time() + time_limit
        iters_left = iterations

        while iters_left > 0 and time.time() < end_time:
            leaf = self._select()
            child = leaf.expand() if not leaf.is_terminal else leaf
            reward, moves = self._simulate(child)
            self._backpropagate(child, reward, moves)
            iters_left -= 1

        # Pick the child with the highest visit count
        best_move, best_child = max(self._root.children.items(), key=lambda c: (c[1].N, c[1].Q))

        # Update to new state
        self._root = best_child
        self._root.parent = None

        r, c = divmod(best_move, DisjointSetBoardV1.N)
        return Move(r, c)

    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root."""
        move = (opp_move.x * DisjointSetBoardV1.N + opp_move.y) if opp_move is not None else None

        # Reuse the tree if possible
        if self._root is not None and move is not None and move in self._root.children:
            self._root = self._root.children[move]
            self._root.parent = None
        # Otherwise, create a completely new root node
        else:
            if opp_move == MCTSV1.SWAP_MOVE:
                self.colour = Colour.opposite(self.colour)
            self._root = MCTSNodeV1(self.colour, DisjointSetBoardV1.from_existing_board(board))

    def _select(self) -> MCTSNodeV1:
        """Find an unexplored descendent of the root node."""
        node = self._root
        while not node.is_terminal and node.is_fully_explored:
            node = self._uct_select(node)
        return node

    # TODO: Which formula for alpha / beta should we use?
    def _uct_select(self, parent: MCTSNodeV1) -> MCTSNodeV1:
        """Select a child of node, balancing exploration & exploitation."""
        def uct_rave(move: int, child: MCTSNodeV1) -> float:
            exploit = child.Q / (child.N + 1e-9)
            explore = MCTSV1.EXPLORATION_WEIGHT * math.sqrt(math.log(parent.N + 1e-9) / (child.N + 1e-9))

            rave_Q, rave_N = parent.rave_Q[move], parent.rave_N[move]
            if rave_N > 0:
                amaf = rave_Q / rave_N
                alpha = max(0.0, (MCTSV1.RAVE_K - child.N) / MCTSV1.RAVE_K)
                return alpha * amaf + (1 - alpha) * exploit + explore
            return exploit + explore # Standard UCT

        return max(parent.children.items(), key=lambda item: uct_rave(item[0], item[1]))[1]

    def _simulate(self, node: MCTSNodeV1) -> tuple[int, list[int]]:
        """Play through the entire game until a winner is found."""
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Play until a winner is found
        simulated_moves = []
        winner = board.check_winner()
        while winner is None:
            move = choice(list(board.possible_moves))
            simulated_moves.append(move)
            winner = board.place(move, current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.check_winner() == self._root.colour else -1, simulated_moves

    @staticmethod
    def _backpropagate(node: MCTSNodeV1, reward: float, moves: list[int]) -> None:
        """Backpropagates rewards and visits until the root node is reached"""
        start_colour = node.colour
        current_node = node
        current_reward = reward
        while current_node is not None:
            # MCTS update
            current_node.Q += current_reward
            current_node.N += 1

            # RAVE update
            # Even indices are moves that the original node made
            # Odd indices were made by the other node
            offset = 0 if current_node.colour == start_colour else 1
            for i in range(offset, len(moves), 2):
                current_node.rave_Q[moves[i]] += current_reward
                current_node.rave_N[moves[i]] += 1

            current_node = current_node.parent
            current_reward = -current_reward # Flip reward as 0-sum
