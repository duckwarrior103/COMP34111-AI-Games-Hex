from src.Board import Board
from src.Colour import Colour


class DisjointSetBoardOld:

    # Board size
    N = 11
    SIZE = N * N

    # Virtual nodes representing the borders
    RED_TOP = SIZE
    RED_BOTTOM = SIZE + 1
    BLUE_LEFT = SIZE + 2
    BLUE_RIGHT = SIZE + 3

    NEIGHBOUR_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    BRIDGE_PATTERNS = [
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (-1, 0, 0, -1),
        (0, -1, -1, 0),
    ]

    # Initialise these after class declaration
    NEIGHBOURS: list[list[int]] | None = None
    BRIDGE_PAIRS: list[list[tuple[int, int]]] | None = None

    def __init__(self):
        self._state = [0] * self.SIZE # A 1D coordinate system
        self._parents = list(range(self.SIZE + 4)) # Each element is initially its own parent
        self._ranks = [0] * (self.SIZE + 4)
        self.possible_moves = set(list(range(self.SIZE))) # Using a set for O(1) removal

    @classmethod
    def from_existing_board(cls, board: Board) -> 'DisjointSetBoardOld':
        """Converts a Board to a DisjointSetBoardOld."""
        dsu_board = cls()
        for r in range(DisjointSetBoardOld.N):
            for c in range(DisjointSetBoardOld.N):
                if board.tiles[r][c].colour is not None:
                    index = r * DisjointSetBoardOld.N + c
                    dsu_board.place(index, board.tiles[r][c].colour)
        return dsu_board

    def get_cell(self, index: int) -> Colour | None:
        """Returns the colour associated with the given index."""
        return Colour.RED if self._state[index] == 1 else Colour.BLUE if self._state[index] == 2 else None

    def place(self, index: int, colour: Colour) -> Colour | None:
        """Places a colour at the index and updates the disjoint sets."""
        colour = 1 if colour == Colour.RED else 2
        self._state[index] = colour

        # Check neighbours and union
        for neighbour_index in self.NEIGHBOURS[index]:
            if self._state[neighbour_index] == self._state[index]:
                self._union(index, neighbour_index)

        # If on the edge, connect to virtual nodes
        r, c = divmod(index, self.N)
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

        self.possible_moves.remove(index)
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
        if self._parents[x] != x:
            self._parents[x] = self._find(self._parents[x])
        return self._parents[x]
    
    def _union(self, x: int, y: int) -> bool:
        x_root = self._find(x)
        y_root = self._find(y)
        
        if x_root == y_root:
            return False
        
        # Union by rank
        if self._ranks[x_root] < self._ranks[y_root]:
            self._parents[x_root] = y_root
        elif self._ranks[x_root] > self._ranks[y_root]:
            self._parents[y_root] = x_root
        else:
            self._parents[y_root] = x_root
            self._ranks[x_root] +=1
        
        return True


# Precompute neighbours
DisjointSetBoardOld.NEIGHBOURS = [
    [
        r * DisjointSetBoardOld.N + c
        for dr, dc in DisjointSetBoardOld.NEIGHBOUR_OFFSETS
        for (r, c) in [(index // DisjointSetBoardOld.N + dr, index % DisjointSetBoardOld.N + dc)]
        if 0 <= r < DisjointSetBoardOld.N and 0 <= c < DisjointSetBoardOld.N
    ]
    for index in range(DisjointSetBoardOld.SIZE)
]

# Precompute bridge pairs
DisjointSetBoardOld.BRIDGE_PAIRS = [
    [
        ((r1 * DisjointSetBoardOld.N + c1), (r2 * DisjointSetBoardOld.N + c2))
        for (dr1, dc1, dr2, dc2) in DisjointSetBoardOld.BRIDGE_PATTERNS
        for (r1, c1, r2, c2) in [
            (
                index // DisjointSetBoardOld.N + dr1,
                index % DisjointSetBoardOld.N + dc1,
                index // DisjointSetBoardOld.N + dr2,
                index % DisjointSetBoardOld.N + dc2
            )
        ]
        if 0 <= r1 < DisjointSetBoardOld.N and 0 <= c1 < DisjointSetBoardOld.N and
           0 <= r2 < DisjointSetBoardOld.N and 0 <= c2 < DisjointSetBoardOld.N
    ]
    for index in range(DisjointSetBoardOld.SIZE)
]