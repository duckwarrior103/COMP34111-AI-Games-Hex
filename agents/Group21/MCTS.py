import copy
import math
import time
from random import choice

from agents.Group21.DisjointSetBoard import DisjointSetBoard
from agents.Group21.MCTSNode import MCTSNode
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTS:

    # Hyperparameters
    EXPLORATION_WEIGHT = 1.0
    RAVE_K = 100

    SWAP_MOVE = (-1, -1)

    BRIDGE_PATTERNS = [
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (-1, 0, 0, -1),
        (0, -1, -1, 0)
    ]

    def __init__(self, colour: Colour):
        self.colour = colour
        self.root: MCTSNode | None = None

    def run(self, time_limit: float = 0.5, iterations: int = 2000) -> Move:
        assert self.root is not None, "Call update(board, opp_move) before run() to set root."

        end_time = time.time() + time_limit
        iters_left = iterations

        while iters_left > 0 and time.time() < end_time:
            leaf = self._select()
            child = leaf.expand() if not leaf.is_terminal else leaf
            reward, moves = self._simulate(child)
            self._backpropagate(child, reward, moves)
            iters_left -= 1

        if not self.root.children:
            r, c = choice(self.root.unexplored_moves)
            return Move(r, c)

        # Pick the child with the highest visit count
        best_move, best_child = max(self.root.children.items(), key=lambda c: (c[1].N, c[1].Q))
        print(f'Best move was {best_move} with Q={best_child.Q}, N={best_child.N}')

        # Update to new state
        self.root = best_child
        self.root.parent = None

        r, c = best_move
        return Move(r, c)

    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root."""
        move = (opp_move.x, opp_move.y) if opp_move is not None else None

        # Reuse the tree if possible
        if self.root is not None and move is not None and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        # Otherwise, create a completely new root node
        else:
            if (move == MCTS.SWAP_MOVE):
                self.colour = Colour.opposite(self.colour)
            self.root = MCTSNode(self.colour, DisjointSetBoard.from_existing_board(board))

    def _select(self) -> MCTSNode:
        """Find an unexplored descendent of the root node."""
        node = self.root
        while not node.is_terminal and node.is_fully_explored:
            node = self._uct_select(node)
        return node

    # TODO: Which formula for alpha / beta should we use?
    def _uct_select(self, parent: MCTSNode) -> MCTSNode:
        """Select a child of node, balancing exploration & exploitation."""
        def uct_rave(move: tuple[int, int], child: MCTSNode) -> float:
            exploit = child.Q / (child.N + 1e-9)
            explore = MCTS.EXPLORATION_WEIGHT * math.sqrt(math.log(parent.N + 1e-9) / (child.N + 1e-9))

            q_rave, n_rave = parent.RAVE[move]
            if n_rave > 0:
                amaf = q_rave / n_rave
                alpha = max(0.0, (MCTS.RAVE_K - child.N) / MCTS.RAVE_K)
                return alpha * amaf + (1 - alpha) * exploit + explore
            return exploit + explore # Standard UCT

        return max(parent.children.items(), key=lambda item: uct_rave(item[0], item[1]))[1]

    def _simulate(self, node: MCTSNode) -> tuple[int, list[tuple[int, int]]]:
        """Play through the entire game until a winner is found."""
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Play until a winner is found
        simulated_moves = []
        winner = board.check_winner()
        while winner is None:
            moves = self._biased_simulation_moves(board, current_colour)
            move = choice(moves)
            simulated_moves.append(move)
            winner = board.place(move[0], move[1], current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.check_winner() == self.root.colour else -1, simulated_moves

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
        scored = [(self._move_heuristic(board, mv, colour), mv) for mv in possible_moves]
        scored.sort(reverse=True)

        # Keep a maximum of the 4 best moves
        return [mv for _, mv in scored[:max(4, len(scored)//5)]]

    def _move_heuristic(self, board: DisjointSetBoard, move: tuple[int, int], colour: Colour) -> float:
        x, y = move
        n = board.N

        opponent_colour = Colour.opposite(self.colour)

        # Compute distance to the target winning side
        if colour == Colour.RED:
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
                if board.get_cell(nx, ny) == colour:
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

            if board.get_cell(x1, y1) == colour and board.get_cell(x2, y2) == colour:
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
    def _backpropagate(node: MCTSNode, reward: float, moves: list[tuple[int, int]]) -> None:
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
                current_node.RAVE[moves[i]][0] += current_reward
                current_node.RAVE[moves[i]][1] += 1

            current_node = current_node.parent
            current_reward = -current_reward # Flip reward as 0-sum
