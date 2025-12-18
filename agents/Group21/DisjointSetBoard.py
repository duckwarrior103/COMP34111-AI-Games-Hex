from src.Board import Board
from src.Colour import Colour


class DisjointSetBoard:

    # Board size
    N = 11
    SIZE = N * N

    # Virtual nodes representing the borders
    RED_TOP = SIZE
    RED_BOTTOM = SIZE + 1
    BLUE_LEFT = SIZE + 2
    BLUE_RIGHT = SIZE + 3

    NEIGHBOUR_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    # Initialise after class declaration
    NEIGHBOURS: list[list[int]] | None = None

    def __init__(self):
        self._state = [0] * self.SIZE # A 1D coordinate system
        self._parents = list(range(self.SIZE + 4)) # Each element is initially its own parent
        self._ranks = [0] * (self.SIZE + 4)

        # Maintain a list of the possible moves, along with a companion array
        # The value at each index in the companion array is the move's actual position within possible_moves
        # Need this for O(1) removal
        self.legal_moves = list(range(self.SIZE))
        self._move_to_index = list(range(self.SIZE))

    @classmethod
    def from_existing_board(cls, board: Board) -> 'DisjointSetBoard':
        """Converts a Board to a DisjointSetBoard."""
        dsu_board = cls()
        for r in range(DisjointSetBoard.N):
            for c in range(DisjointSetBoard.N):
                if board.tiles[r][c].colour is not None:
                    index = r * DisjointSetBoard.N + c
                    dsu_board.place(index, board.tiles[r][c].colour)
        return dsu_board

    def get_cell(self, index: int) -> Colour | None:
        """Returns the colour associated with the given index."""
        return Colour.RED if self._state[index] == 1 else Colour.BLUE if self._state[index] == 2 else None

    def place(self, index: int, colour: Colour) -> None:
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

        self._remove_move(index)

    def check_winner(self) -> Colour | None:
        """Check if a winner exists efficiently by checking disjoint sets."""
        if self._find(self.RED_TOP) == self._find(self.RED_BOTTOM):
            return Colour.RED
        elif self._find(self.BLUE_LEFT) == self._find(self.BLUE_RIGHT):
            return Colour.BLUE
        else:
            return None

    def find_forced_move(self, colour: Colour) -> int | None:
        """Find a forced move, if any, for the given colour."""
        for move in self.legal_moves:
            if self.is_winning_move(move, colour):
                return move

        # Can be more than one losing move that we need to block, but here we just return the first we find
        # Because if there's more than one we're cooked anyway
        for move in self.legal_moves:
            if self.is_winning_move(move, Colour.opposite(colour)):
                return move

        return None

    def is_winning_move(self, index: int, colour: Colour) -> bool:
        """Does placing this move here with the given colour result in an immediate win?"""
        colour = 1 if colour == Colour.RED else 2

        # Check if current move is on the border
        r, c = divmod(index, self.N)
        if colour == 1:
            edge1_root, edge2_root = self._find(self.RED_TOP), self._find(self.RED_BOTTOM)
            touches_edge1 = (r == 0)
            touches_edge2 = (r == self.N - 1)
        else:
            edge1_root, edge2_root = self._find(self.BLUE_LEFT), self._find(self.BLUE_RIGHT)
            touches_edge1 = (c == 0)
            touches_edge2 = (c == self.N - 1)

        # Check neighbours of the same colour
        for neighbour in self.NEIGHBOURS[index]:
            if self._state[neighbour] == colour:
                root = self._find(neighbour)
                if root == edge1_root:
                    touches_edge1 = True
                if root == edge2_root:
                    touches_edge2 = True

                # Return early if this move means it will connect the two edges
                if touches_edge1 and touches_edge2:
                    return True

        # Iterated over the neighbours and never returned True, so this must not be a winning move
        return False

    def _remove_move(self, move: int) -> None:
        """Efficient removal of a move in O(1) time."""
        move_index = self._move_to_index[move]
        last_index = len(self.legal_moves) - 1
        last_move = self.legal_moves[last_index]
        self.legal_moves[move_index] = last_move
        self._move_to_index[last_move] = move_index
        self.legal_moves.pop()
        self._move_to_index[move] = -1 # Mark as removed

    # Disjoint Union Set Functions
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
DisjointSetBoard.NEIGHBOURS = [
    [
        r * DisjointSetBoard.N + c
        for dr, dc in DisjointSetBoard.NEIGHBOUR_OFFSETS
        for (r, c) in [(index // DisjointSetBoard.N + dr, index % DisjointSetBoard.N + dc)]
        if 0 <= r < DisjointSetBoard.N and 0 <= c < DisjointSetBoard.N
    ]
    for index in range(DisjointSetBoard.SIZE)
]
