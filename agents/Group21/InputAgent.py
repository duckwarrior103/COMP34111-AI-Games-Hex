from random import choice, random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class InputAgent(AgentBase):
    """
    We can play against this agent by providing input in the console.
    """

    _board_size: int = 11
    _choices: list[tuple[int, int]]

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Valid moves will hold all our valid moves
        valid_moves = [
            (i, j)
            for i in range(board.size)
            for j in range(board.size)
            if not board.tiles[i][j].colour
        ]

        # print("Current Board:")
        # print(board)
        # print(f"Valid moves: {valid_moves}")


        # We want the user to keep on going until they make a valid move
        while True:
            x,y = map(int, input("take a move (format y x):").split())
            user_move = (x,y)
            print(user_move)
            if user_move in valid_moves:
                return Move(x,y)
