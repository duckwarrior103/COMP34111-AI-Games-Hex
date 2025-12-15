import copy

from agents.Group21.DisjointSetBoard import DisjointSetBoard
from agents.Group21.old.DisjointSetBoardOld import DisjointSetBoardOld
from src.Colour import Colour
from random import randrange


class MCTSNodeOld:
    def __init__(
        self, 
        colour: Colour,
        board: DisjointSetBoardOld,
        parent: "MCTSNodeOld | None" = None
    ):
        self.board = board
        self.colour = colour
        self.parent = parent

        self.children: dict[int, MCTSNodeOld] = {} # Move -> MCTSNode
        self.Q = 0.0 # Total reward
        self.N = 0 # Total number of visits
        self.rave_Q = [0.0] * DisjointSetBoardOld.SIZE
        self.rave_N = [0] * DisjointSetBoardOld.SIZE

        self.unexplored_moves = list(self.board.possible_moves)
        self.is_terminal = self.board.check_winner() is not None

    @property
    def is_fully_explored(self) -> bool:
        """Returns True if this all children of this node have been explored at least once"""
        return not self.unexplored_moves

    # TODO: Currently not using Gan's heuristic
    def expand(self) -> 'MCTSNodeOld':
        """
        Selects a random unexplored move, removing it from the list
        before creating the child MCTSNode and adding to self.children.
        """
        random_index = randrange(len(self.unexplored_moves))
        move = self.unexplored_moves[random_index]

        # Swap with last element and pop for O(1) removal
        # Not sure how much difference this actually makes
        last_index = len(self.unexplored_moves) - 1
        if random_index != last_index:
            self.unexplored_moves[random_index] = self.unexplored_moves[last_index]
        self.unexplored_moves.pop()

        board_copy = copy.deepcopy(self.board)
        board_copy.place(move, self.colour)

        child_node = MCTSNodeOld(
            colour=Colour.opposite(self.colour),
            board=board_copy,
            parent=self
        )
        self.children[move] = child_node
        return child_node
