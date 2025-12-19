import copy
import random

from agents.Group21.v3.DisjointSetBoardV3 import DisjointSetBoardV3
from src.Colour import Colour


class MCTSNodeV3:
    def __init__(
        self, 
        colour: Colour,
        board: DisjointSetBoardV3,
        parent: "MCTSNodeV3 | None" = None,
        prev_move: int | None = None,
    ):
        self.board = board
        self.colour = colour
        self.parent = parent
        self.prev_move = prev_move

        self.children: dict[int, MCTSNodeV3] = {} # Move -> MCTSNodeV3
        self.W = 0.0 # Total reward
        self.N = 0 # Total number of visits
        self.rave_W = [0.5] * DisjointSetBoardV3.SIZE # Neutral prior
        self.rave_N = [8] * DisjointSetBoardV3.SIZE

        # Copy the list of possible moves and shuffle
        # Allows for us to pop from the end to get a random move -> O(1)
        self.unexplored_moves = self.board.legal_moves[:]
        random.shuffle(self.unexplored_moves)

        self.is_terminal = self.board.check_winner() is not None

    @property
    def is_fully_explored(self) -> bool:
        """Returns True if this all children of this node have been explored at least once"""
        return not self.unexplored_moves

    def expand(self) -> 'MCTSNodeV3':
        """
        Searches first for any forced moves (immediate wins, losses or save-bridges).
        If so, we return that move and prune any siblings.
        Otherwise, we expand with a random unexplored node.
        """
        forced_move = self.board.find_forced_move(self.colour, self.prev_move)
        if forced_move:
            move = forced_move
            self.unexplored_moves.clear()
        else:
            move = self.unexplored_moves.pop()

        board_copy = copy.deepcopy(self.board)
        board_copy.place(move, self.colour)

        child_node = MCTSNodeV3(
            colour=Colour.opposite(self.colour),
            board=board_copy,
            parent=self,
            prev_move=move,
        )
        self.children[move] = child_node
        return child_node
