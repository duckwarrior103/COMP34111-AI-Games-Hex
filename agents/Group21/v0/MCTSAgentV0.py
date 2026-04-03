from agents.Group21.v0.MCTSV0 import MCTSV0
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSAgentV0(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.mcts = MCTSV0(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Update MCTS with opponents move, select a move using MCTS and then update again
        self.mcts.update(board, opp_move)
        move = self.mcts.run()
        self.mcts.update(board, move)
        return move
