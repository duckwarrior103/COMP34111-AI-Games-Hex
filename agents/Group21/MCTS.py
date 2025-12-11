import copy
import math
import time
from random import choice

from agents.Group21.disjoint_set_board import DisjointSetBoard
from agents.Group21.MCTSNode import MCTSNode
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTS:
    BRIDGE_PATTERNS = [
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (-1, 0, 0, -1),
        (0, -1, -1, 0)
    ]

    def __init__(self, colour: Colour, exploration_weight: float = 1):
        self.colour = colour
        self.root: MCTSNode | None = None
        self.exploration_weight = exploration_weight

    def run(self, time_limit: float = 0.5, iterations: int = 2000) -> Move:
        assert self.root is not None, "Call update(board, opp_move) before run() to set root."

        end_time = time.time() + time_limit
        iters_left = iterations

        while iters_left > 0 and time.time() < end_time:
            leaf = self._select()
            child = leaf.expand() if not leaf.is_terminal else leaf
            reward = self._simulate(child)
            self._backpropagate(child, reward)
            iters_left -= 1

        if not self.root.children:
            r, c = choice(self.root.unexplored_moves)
            return Move(r, c)

        # Pick the child with the highest visit count
        best_move, best_child = max(self.root.children.items(), key=lambda c: c[1].N)
        print(f'Best move was {best_move} with Q={best_child.Q}, N={best_child.N}')
        r, c = best_move
        return Move(r, c)

    # TODO: Moving to the next move currently results in worse performance than starting anew
    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root."""
        # opp_move_idx = DisjointSetBoard.coords_to_index(opp_move.x, opp_move.y) if opp_move is not None else None
        move = (opp_move.x, opp_move.y) if opp_move is not None else None

        # Reuse the tree if possible
        if self.root is not None and move is not None and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        # Otherwise, create a completely new root node
        else:
            self.root = MCTSNode(self.colour, DisjointSetBoard.from_existing_board(board))

    def _select(self) -> MCTSNode:
        """Find an unexplored descendent of the root node."""
        node = self.root
        while not node.is_terminal and node.is_fully_explored:
            node = self._uct_select(node)
        return node

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """Select a child of node, balancing exploration & exploitation."""
        log_N_vertex = math.log(node.N + 1e-9)

        def uct(n: MCTSNode) -> float:
            """Returns the upper confidence bound for trees."""
            return (n.Q / (n.N + 1e-9)) + self.exploration_weight * math.sqrt(log_N_vertex / (n.N + 1e-9))

        return max(node.children.values(), key=uct)

    def _simulate(self, node: MCTSNode) -> float:
        """Play through the entire game until a winner is found."""
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Play until a winner is found
        winner = board.check_winner()
        while winner is None:
            moves = self._biased_simulation_moves(board, current_colour)
            move = choice(moves)
            winner = board.place(move[0], move[1], current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.check_winner() == self.root.colour else -1

    def _biased_simulation_moves(self, board: DisjointSetBoard, colour: Colour) -> list[tuple[int, int]]:
        n = board.N
        possible_moves = board.possible_moves

        # Prefer moves adjacent to existing own color
        good = []
        for x, y in possible_moves:
            for dx, dy in board.NEIGHBOUR_OFFSETS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if board.get_cell(nx, ny) == colour:
                        good.append((x, y))
                        break

        if good:
            return good

        # If no adjacent moves exist, use heuristic scoring
        scored = [(self._move_heuristic(board, mv), mv) for mv in possible_moves]
        scored.sort(reverse=True)

        # Keep a maximum of the 4 best moves
        return [mv for _, mv in scored[:max(4, len(scored)//5)]]

    def _move_heuristic(self, board: DisjointSetBoard, move: Move | tuple[int, int]) -> float:
        if isinstance(move, Move):
            x, y = move.x, move.y
        else:
            x, y = move

        n = board.N
        opponent_colour = Colour.opposite(self.colour)

        # Compute distance to the target winning side
        if self.colour == Colour.RED:
            dist_goal = min(y, n - 1 - y)
            goal_axis = 1  # y-direction
        else:
            dist_goal = min(x, n - 1 - x)
            goal_axis = 0  # x-direction

        # Computes center preference (the more center the better)
        center_score = -((x - n / 2) ** 2 + (y - n / 2) ** 2)

        # When adjacent is same color
        adj_bonus = 0
        for dx, dy in DisjointSetBoard.NEIGHBOUR_OFFSETS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                if board.get_cell(nx, ny) == self.colour:
                    adj_bonus += 3
                if board.get_cell(nx, ny) == opponent_colour:
                    adj_bonus -= 1

        # Diagonal cells, where empty space in between is almost impossible for opponent to break
        bridge_bonus = 0
        for dx, dy, ex, ey in self.BRIDGE_PATTERNS:
            x1, y1 = x + dx, y + dy
            x2, y2 = x + ex, y + ey

            if not (0 <= x1 < n and 0 <= y1 < n and
                    0 <= x2 < n and 0 <= y2 < n):
                continue

            if board.get_cell(x1, y1) == self.colour and board.get_cell(x2, y2) == self.colour:
                if goal_axis == 1:
                    if y1 != y2:
                        bridge_bonus += 6
                else:
                    if x1 != x2:
                        bridge_bonus += 6

        return (
            -2 * dist_goal +
            adj_bonus +
            bridge_bonus +
            0.15 * center_score
        )

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float):
        """Backpropagates rewards and visits until the root node is reached"""
        current_node = node
        current_reward = reward
        while current_node is not None:
            current_node.Q += current_reward
            current_node.N += 1

            current_node = current_node.parent
            current_reward = -current_reward # Flip reward as 0-sum
