from agents.Group21.v1.MCTSV1 import MCTSV1
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSAgentV1(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.mcts = MCTSV1(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        self.mcts.update(board, opp_move)
        move = self.mcts.run()
        return move
