import copy

from agents.Group21.disjoint_set_board import DisjointSetBoard
from src.Colour import Colour
from random import randrange


class MCTSNode():
    def __init__(
        self, 
        colour: Colour,
        board: DisjointSetBoard,
        parent: "MCTSNode | None" = None
    ):
        self.board = board
        self.colour = colour
        self.parent = parent

        self.children: dict[int, MCTSNode] = {} # Key is 1D coordinate for move, value is the Node it leads to
        self.Q = 0 # Total reward
        self.N = 0 # Total number of visits

        self.unexplored_moves = list(self.board.possible_moves)
        self.is_terminal = self.board.check_winner() is not None

    @property
    def is_fully_explored(self) -> bool:
        """Returns True if this all children of this node have been explored at least once"""
        return not self.unexplored_moves

    # TODO: Removing from middle of the list is O(n), could remove from the end instead if further optimisation needed
    def expand(self) -> 'MCTSNode':
        """
        Selects a random unexplored move, removing it from the list
        before creating the child MCTSNode and adding to self.children.
        """
        random_index = randrange(len(self.unexplored_moves))
        move_index = self.unexplored_moves[random_index]

        last_index = len(self.unexplored_moves) - 1
        if random_index != last_index:
            self.unexplored_moves[random_index] = self.unexplored_moves[last_index]
        self.unexplored_moves.pop()

        # Convert to 2D indices
        r, c = DisjointSetBoard.index_to_coords(move_index)

        board_copy = copy.deepcopy(self.board)
        board_copy.place(r, c, self.colour)

        child_node = MCTSNode(
            colour=Colour.opposite(self.colour),
            board=board_copy,
            parent=self
        )
        self.children[move_index] = child_node
        return child_node