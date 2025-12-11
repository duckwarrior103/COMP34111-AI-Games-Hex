import random
from selectors import SelectSelector

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


# TODO: Implement swap
class DisjointSetBoard:
    N = 11
    SIZE = N * N

    # Virtual nodes representing the borders
    RED_TOP = SIZE
    RED_BOTTOM = SIZE + 1
    BLUE_LEFT = SIZE + 2
    BLUE_RIGHT = SIZE + 3

    NEIGHBOUR_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    def __init__(self):
        self.state = [0] * self.SIZE
        self.possible_moves = set(range(self.SIZE)) # Set for O(1) removal
        self.parents = list(range(self.SIZE + 4)) # Each element is initially its own parent
        self.ranks = [0] * (self.SIZE + 4)

    @classmethod
    def from_existing_board(cls, board: Board) -> 'DisjointSetBoard':
        """Converts a Board to a DisjointSetBoard"""
        dsu_board = cls()
        for r in range(DisjointSetBoard.N):
            for c in range(DisjointSetBoard.N):
                if board.tiles[r][c].colour is not None:
                    dsu_board.place(r, c, board.tiles[r][c].colour)
        return dsu_board

    def get_cell(self, r: int, c: int) -> Colour | None:
        pos = (r * self.N) + c
        return Colour.RED if self.state[pos] == 1 else Colour.BLUE if self.state[pos] == 2 else None

    def place(self, r: int, c: int, colour: Colour) -> Colour | None:
        colour = 1 if colour == Colour.RED else 2

        # Update state
        pos = (r * self.N) + c
        self.state[pos] = colour

        # Check neighbours and union
        for dr, dc in self.NEIGHBOUR_OFFSETS:
            i, j = r + dr, c + dc

            if 0 <= i < self.N and 0 <= j < self.N:
                neighbour_pos = (i * self.N) + j
                if self.state[neighbour_pos] == self.state[pos]:
                    self._union(pos, neighbour_pos)

        # If on the edge, connect to virtual nodes
        if colour == 1:
            if r == 0:
                self._union(pos, self.RED_TOP)
            elif r == self.N - 1:
                self._union(pos, self.RED_BOTTOM)
        else:
            if c == 0:
                self._union(pos, self.BLUE_LEFT)
            elif c == self.N - 1:
                self._union(pos, self.BLUE_RIGHT)

        self.possible_moves.remove(pos)
        return self.check_winner()

    def check_winner(self) -> Colour | None:
        if self._find(self.RED_TOP) == self._find(self.RED_BOTTOM):
            return Colour.RED
        elif self._find(self.BLUE_LEFT) == self._find(self.BLUE_RIGHT):
            return Colour.BLUE
        else:
            return None

    def _find(self, x: int) -> int:
        # Path compression
        if self.parents[x] != x:
            self.parents[x] = self._find(self.parents[x])
        return self.parents[x]
    
    def _union(self, x: int, y: int) -> bool:
        x_root = self._find(x)
        y_root = self._find(y)
        
        if x_root == y_root:
            return False
        
        # Union by rank
        if self.ranks[x_root] < self.ranks[y_root]:
            self.parents[x_root] = y_root
        elif self.ranks[x_root] > self.ranks[y_root]:
            self.parents[y_root] = x_root
        else:
            self.parents[y_root] = x_root
            self.ranks[x_root] +=1
        
        return True

    @staticmethod
    def index_to_coords(index: int) -> tuple[int, int]:
        return index // DisjointSetBoard.N, index % DisjointSetBoard.N
