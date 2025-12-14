from pathlib import Path

import torch
from agents.Group21.ZeroMCTS import ZeroMCTS
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import numpy as np

class ZeroAgent(AgentBase):
    def __init__(self, colour: Colour, simulations=50):
        self.colour = colour

        def load_model():
            model_name="hex_neural_net.pth"
            # Project root (two levels up from this script)
            project_root = Path(__file__).resolve().parents[2]
            models_dir = project_root / "saved_models"
            model_path = models_dir / model_name   
            # Load the full saved model
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            print("Neural network model loaded successfully.")
            return model

        neural_net = load_model()
        self.mcts = ZeroMCTS(neural_net)
        self.simulations = simulations

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        legal_moves, pi = self.mcts.run(board, simulations=self.simulations)
        # Select move_index with highest probability in pi distribution
        move_index = np.argmax(pi)
        print(legal_moves[move_index])
        x, y = legal_moves[move_index]
        
        move = Move(x, y)
        print(isinstance(move, Move))

        return move