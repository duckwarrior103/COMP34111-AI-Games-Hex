import copy
import math
import time
from random import choice

from agents.Group21.v0.MCTSNodeV0 import MCTSNodeV0
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSV0:
    def __init__(self, colour: Colour, exploration_weight: float = 1):
        self.colour = colour
        self.root: MCTSNodeV0 | None = None
        self.exploration_weight = exploration_weight

    def run(self, time_limit: float = 0.5, iterations: int = 2000) -> Move:
        assert self.root is not None, "Call update(board, opp_move) before run() to set root."

        end_time = time.time() + time_limit
        iters_left = iterations

        while iters_left > 0 and time.time() < end_time:
            leaf = self._select()
            child = self._expand(leaf) if not leaf.is_terminal else leaf
            reward = self._simulate(child)
            self._backpropagate(child, reward)
            iters_left -= 1

        # Picking the child with the highest visit count
        best_move, best_child = max(self.root.children.items(), key=lambda child: child[1].N)
        return best_move

    # TODO: Should we prune the old states? Do we need to keep track?
    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root"""

        # Initial set up
        if self.root is None:
            self.root = MCTSNodeV0(self.colour, board)
            return

        # Check if we have a node for the opponent's move, in which case we can reuse
        if opp_move in self.root.children:
            self.root = self.root.children[opp_move]
            self.root.parent = None
            return

        self.root = MCTSNodeV0(self.colour, board)

    def _select(self) -> MCTSNodeV0:
        """Find an unexplored descendent of the root node"""
        node = self.root
        while not node.is_terminal and node.is_fully_explored:
            node = self._uct_select(node)
        return node

    # TODO: Division by 0 for N = 0?
    def _uct_select(self, node: MCTSNodeV0) -> MCTSNodeV0:
        """Select a child of node, balancing exploration & exploitation"""
        log_N_vertex = math.log(node.N + 1e-9)

        def uct(n: MCTSNodeV0) -> float:
            """Returns the upper confidence bound for trees"""
            return (n.Q / (n.N + 1e-9)) + self.exploration_weight * math.sqrt(log_N_vertex / (n.N + 1e-9))

        return max(node.children.values(), key=uct)

    def _expand(self, node: MCTSNodeV0) -> MCTSNodeV0:
        move = choice(node.unexplored_moves)
        return node.make_move(move)

    def _simulate(self, node: MCTSNodeV0) -> float:
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        while not board.has_ended(Colour.opposite(current_colour)):
            empty = [
                (i, j)
                for i in range(board.size)
                for j in range(board.size)
                if not board.tiles[i][j].colour
            ]
            move = choice(empty)
            board.set_tile_colour(move[0], move[1], current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.get_winner() == self.root.colour else -1

    @staticmethod
    def _backpropagate(node: MCTSNodeV0, reward: float):
        """Backpropagates rewards and visits until the root node is reached"""
        current_node = node
        current_reward = reward
        while current_node is not None:
            current_node.Q += current_reward
            current_node.N += 1

            current_node = current_node.parent
            current_reward = -current_reward  # Flip reward as 0-sum