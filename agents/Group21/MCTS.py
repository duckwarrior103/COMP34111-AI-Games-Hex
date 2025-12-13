import copy
import math
import time
from random import choice, shuffle

from agents.Group21.DisjointSetBoard import DisjointSetBoard
from agents.Group21.MCTSNode import MCTSNode
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTS:

    # Hyperparameters
    EXPLORATION_WEIGHT = 1.0
    RAVE_K = 100

    SWAP_MOVE = -12 # (-1, -1) maps to -12

    def __init__(self, colour: Colour):
        self.colour = colour
        self._root: MCTSNode | None = None

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

        if not self._root.children:
            move = choice(self._root.unexplored_moves)
            r, c = divmod(move, DisjointSetBoard.N)
            return Move(r, c)

        # Pick the child with the highest visit count
        best_move, best_child = max(self._root.children.items(), key=lambda c: (c[1].N, c[1].Q))

        # Update to new state
        self._root = best_child
        self._root.parent = None

        r, c = divmod(best_move, DisjointSetBoard.N)
        return Move(r, c)

    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root."""
        move = (opp_move.x * DisjointSetBoard.N + opp_move.y) if opp_move is not None else None

        # Reuse the tree if possible
        if self._root is not None and move is not None and move in self._root.children:
            self._root = self._root.children[move]
            self._root.parent = None
        # Otherwise, create a completely new root node
        else:
            if opp_move == MCTS.SWAP_MOVE:
                self.colour = Colour.opposite(self.colour)
            self._root = MCTSNode(self.colour, DisjointSetBoard.from_existing_board(board))

    def _select(self) -> MCTSNode:
        """Find an unexplored descendent of the root node."""
        node = self._root
        while not node.is_terminal and node.is_fully_explored:
            node = self._select_best_child(node)
        return node

    @staticmethod
    def _select_best_child(parent: MCTSNode) -> MCTSNode:
        """Select the best child node based off UCT-RAVE."""
        log_parent_N = math.log(parent.N)
        def uct_rave(move: int, child: MCTSNode) -> float:
            exploit = child.Q / child.N
            explore = MCTS.EXPLORATION_WEIGHT * math.sqrt(log_parent_N / child.N)

            rave_Q, rave_N = parent.rave_Q[move], parent.rave_N[move]
            amaf = rave_Q / rave_N
            alpha = max(0.0, (MCTS.RAVE_K - child.N) / MCTS.RAVE_K)
            return alpha * amaf + (1 - alpha) * exploit + explore

        return max(parent.children.items(), key=lambda item: uct_rave(item[0], item[1]))[1]

    def _simulate(self, node: MCTSNode) -> tuple[int, list[int]]:
        """Do full board simulation."""
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Play randomly until the board is full
        moves_to_play = list(board.possible_moves)
        shuffle(moves_to_play)
        for move in moves_to_play:
            board.place(move, current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.check_winner() == self._root.colour else 0, moves_to_play

    def _biased_simulation_moves(self, board: DisjointSetBoard, colour: Colour) -> list[int]:
        possible_moves = board.possible_moves

        # Prefer moves adjacent to existing own color
        good = []
        for move in possible_moves:
            for neighbour in DisjointSetBoard.NEIGHBOURS[move]:
                if (board.get_cell(neighbour)) == colour:
                    good.append(move)
                    break

        if good:
            return good

        # If no adjacent moves exist, use heuristic scoring
        scored = [(self._move_heuristic(board, move, colour), move) for move in possible_moves]
        scored.sort(reverse=True)

        # Keep a maximum of the 4 best moves
        return [move for _, move in scored[:max(4, len(scored)//5)]]

    def _move_heuristic(self, board: DisjointSetBoard, move: int, colour: Colour) -> float:
        x, y = divmod(move, board.N)
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
        for neighbour in DisjointSetBoard.NEIGHBOURS[move]:
            if board.get_cell(neighbour) == colour:
                adj_bonus += 3
            if board.get_cell(neighbour) == opponent_colour:
                adj_bonus -= 1

        # Diagonal cells, where empty space in between is almost impossible for opponent to break
        bridge_bonus = 0
        for move1, move2 in DisjointSetBoard.BRIDGE_PAIRS[move]:
            r1, c1 = divmod(move1, board.N)
            r2, c2 = divmod(move2, board.N)

            if board.get_cell(move1) == colour and board.get_cell(move2) == colour:
                if goal_axis == 1 and r1 != r2:
                    bridge_bonus += 6
                if goal_axis == 0 and c1 != c2:
                    bridge_bonus += 6

        return (
            -2 * dist_goal +
            adj_bonus +
            bridge_bonus +
            0.15 * center_score
        )

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float, moves: list[int]) -> None:
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
