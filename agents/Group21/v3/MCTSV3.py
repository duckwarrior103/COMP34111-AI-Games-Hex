import copy
import math
import random
import time

from agents.Group21.v3.MCTSNodeV3 import MCTSNodeV3
from agents.Group21.v3.DisjointSetBoardV3 import DisjointSetBoardV3
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSV3:

    # Hyperparameters
    EXPLORATION_WEIGHT = 0.5
    RAVE_K = 500
    TIME_LIMIT = 1.0

    RED_OPENINGS = [(1, 2), (1, 8), (9, 2), (9, 8)]
    CENTRE = 60 # (5, 5) -> 60
    SWAP_MOVE = -12 # (-1, -1) -> -12

    def __init__(self, colour: Colour):
        self.colour = colour
        self._root: MCTSNodeV3 | None = None
        self._swapped = False

    def run(self) -> Move:
        """Run the MCTSV3 algorithm, returning the best move."""
        turn = DisjointSetBoardV3.SIZE - len(self._root.board.legal_moves)

        # 1. An opening move - play deterministically from our opening book
        if turn < 2:
            return self._opening_book(turn)

        # 2. Check for a forced move, in which case we can skip MCTSV3 altogether and just play that
        forced_move = self._root.board.find_forced_move(self._root.colour, self._root.prev_move)
        if forced_move:
            r, c = divmod(forced_move, DisjointSetBoardV3.N)
            return Move(r, c)

        # 2. Otherwise, run MCTSV3 to determine the best move
        end_time = time.time() + self.TIME_LIMIT

        while time.time() < end_time:
            leaf = self._select()
            child = leaf.expand() if not leaf.is_terminal else leaf
            winner, moves = self._simulate(child)
            self._backpropagate(child, winner, moves)

        if not self._root.children:
            move = random.choice(self._root.unexplored_moves)
            r, c = divmod(move, DisjointSetBoardV3.N)
            return Move(r, c)

        # Pick the child with the highest visit count
        best_move, best_child = max(self._root.children.items(), key=lambda c: (c[1].N, c[1].W))

        # Update to new state
        self._root = best_child
        self._root.parent = None

        r, c = divmod(best_move, DisjointSetBoardV3.N)
        return Move(r, c)

    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root."""
        move = (opp_move.x * DisjointSetBoardV3.N + opp_move.y) if opp_move is not None else None

        # Reuse the tree if possible
        if self._root is not None and move is not None and move in self._root.children:
            self._root = self._root.children[move]
            self._root.parent = None
        # Otherwise, create a completely new root node
        else:
            # Reverse colours if move was swap
            if move == MCTSV3.SWAP_MOVE:
                self._swapped = True
                self.colour = Colour.opposite(self.colour)
                self._root = MCTSNodeV3(self.colour, DisjointSetBoardV3.from_existing_board(board), prev_move=self._root.prev_move)
            else:
                self._root = MCTSNodeV3(self.colour, DisjointSetBoardV3.from_existing_board(board), prev_move=move)

    def _opening_book(self, turn: int) -> Move | None:
        """A naive opening book for rounds 1 and 2."""
        # Red - take a fair opening that blue will never 100% swap with
        if turn == 0:
            return Move(*random.choice(self.RED_OPENINGS))
        # Blue - two possible scenarios
        elif turn == 1:
            # Other person has swapped with us, take centre guaranteed as we never play centre
            if self._swapped:
                return Move(5, 5)
            # We are true Blue
            else:
                # Swap moves if the move is in the centre
                if self._root.prev_move == self.CENTRE or self._root.prev_move in DisjointSetBoardV3.NEIGHBOURS[self.CENTRE]:
                    return Move(-1, -1)
                # Otherwise take centre ourselves
                else:
                    return Move(5, 5)
        else:
            raise ValueError("No openings available for turns >= 2")

    def _select(self) -> MCTSNodeV3:
        """Find an unexplored descendent of the root node."""
        node = self._root
        while not node.is_terminal and node.is_fully_explored:
            node = self._select_best_child(node)
        return node

    @staticmethod
    def _select_best_child(parent: MCTSNodeV3) -> MCTSNodeV3:
        """Select the best child node based off UCT-RAVE."""
        log_parent_N = math.log(parent.N)
        def uct_rave(move: int, child: MCTSNodeV3) -> float:
            exploit = child.W / child.N
            explore = MCTSV3.EXPLORATION_WEIGHT * math.sqrt(log_parent_N / child.N)

            rave_W, rave_N = parent.rave_W[move], parent.rave_N[move]
            amaf = rave_W / rave_N
            beta = math.sqrt(MCTSV3.RAVE_K / (3 * child.N + MCTSV3.RAVE_K))
            return beta * amaf + (1 - beta) * exploit + explore

        return max(parent.children.items(), key=lambda item: uct_rave(item[0], item[1]))[1]

    def _simulate(self, node: MCTSNodeV3) -> tuple[Colour, list[int]]:
        """Do full board simulation."""
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Play randomly until the board is full
        moves_to_play = board.legal_moves[:]
        random.shuffle(moves_to_play)
        for move in moves_to_play:
            board.place(move, current_colour)
            current_colour = Colour.opposite(current_colour)

        return board.check_winner(), moves_to_play

    @staticmethod
    def _backpropagate(node: MCTSNodeV3, winner: Colour, moves: list[int]) -> None:
        """Backpropagates rewards and visits until the root node is reached"""
        start_colour = node.colour
        current_node = node
        while current_node is not None:
            # MCTSV3 update
            if current_node.colour != winner:
                current_node.W += 1
            current_node.N += 1

            # RAVE update
            # Even indices are moves that the original node made
            # Odd indices were made by the other node
            offset = 0 if current_node.colour == start_colour else 1
            for i in range(offset, len(moves), 2):
                if current_node.colour == winner:
                    current_node.rave_W[moves[i]] += 1
                current_node.rave_N[moves[i]] += 1

            current_node = current_node.parent
