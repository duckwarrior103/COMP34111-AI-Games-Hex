from src.Board import Board
from src.Colour import Colour


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
        self.state = [0] * self.SIZE # A 1D coordinate system
        self.parents = list(range(self.SIZE + 4)) # Each element is initially its own parent
        self.ranks = [0] * (self.SIZE + 4)

        # Using a set for O(1) removal
        self.possible_moves = set(
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
        )

    @classmethod
    def from_existing_board(cls, board: Board) -> 'DisjointSetBoard':
        """Converts a Board to a DisjointSetBoard."""
        dsu_board = cls()
        for r in range(DisjointSetBoard.N):
            for c in range(DisjointSetBoard.N):
                if board.tiles[r][c].colour is not None:
                    dsu_board.place(r, c, board.tiles[r][c].colour)
        return dsu_board

    @staticmethod
    def index_to_coords(index: int) -> tuple[int, int]:
        """Converts a 1D DisjointSetBoard index to its 2D coordinate (r, c) form."""
        return divmod(index, index)

    @staticmethod
    def coords_to_index(r: int, c: int) -> int:
        """Converts a 2D coordinate (r, c) form to a 1D DisjointSetBoard index."""
        return (r * DisjointSetBoard.N) + c

    def get_cell(self, r: int, c: int) -> Colour | None:
        """Returns the colour associated with the cell at (r,c)."""
        index = self.coords_to_index(r, c)
        return Colour.RED if self.state[index] == 1 else Colour.BLUE if self.state[index] == 2 else None

    def place(self, r: int, c: int, colour: Colour) -> Colour | None:
        """
        Places a colour at (r,c) and updates the disjoint sets.

        :return: The winning colour (if one exists)
        """
        colour = 1 if colour == Colour.RED else 2

        index = self.coords_to_index(r, c)
        self.state[index] = colour

        # Check neighbours and union
        for dr, dc in self.NEIGHBOUR_OFFSETS:
            i, j = r + dr, c + dc

            if 0 <= i < self.N and 0 <= j < self.N:
                neighbour_index = self.coords_to_index(i, j)
                if self.state[neighbour_index] == self.state[index]:
                    self._union(index, neighbour_index)

        # If on the edge, connect to virtual nodes
        if colour == 1:
            if r == 0:
                self._union(index, self.RED_TOP)
            elif r == self.N - 1:
                self._union(index, self.RED_BOTTOM)
        else:
            if c == 0:
                self._union(index, self.BLUE_LEFT)
            elif c == self.N - 1:
                self._union(index, self.BLUE_RIGHT)

        self.possible_moves.remove((r, c))
        return self.check_winner()

    def check_winner(self) -> Colour | None:
        """
        Check if a winner exists efficiently by checking disjoint sets.

        :return: The winning Colour or None if no winner.
        """
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
