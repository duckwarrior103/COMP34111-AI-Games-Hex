# generate_selfplay_data.py
import torch
import pickle
from pyexpat import model
from agents.Group21.SelfPlay import SelfPlay
from src.Game import Game
import numpy as np
from pathlib import Path
import time
import argparse
    
def load_model(model_name="hex_neural_net.pth"):
    # Project root (two levels up from this script)
    project_root = Path(__file__).resolve().parents[0]
    models_dir = project_root / "saved_models"
    model_path = models_dir / model_name

    # Fallback to legacy location
    if not model_path.exists():
        alt_path = project_root / "network_dev" / "saved_models" / model_name
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found: {model_path} (tried fallback {alt_path})")

    # Load the full saved model
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return model

def main():
    alpha = "hex_neural_net.pth"

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument('-n', '--num-games', type=int, default=10, help='number of games to generate')
    args = parser.parse_args()

    # Load model
    nn = load_model(model_name=alpha)

    # Initialize SelfPlay engine
    self_play_engine = SelfPlay(
        neural_net=nn,
        game_cls=Game,
        simulations=50
    )

    # Use command-line value for number of games
    num_games = args.num_games
    training_examples = []

    start_time = time.perf_counter()

    for i in range(num_games):
        game_start = time.perf_counter()
        print(f"Playing game {i+1}/{num_games}...")

        examples = self_play_engine.play_game()
        training_examples += examples

        game_time = time.perf_counter() - game_start
        print(f"  Game {i+1} took {game_time:.2f}s")

    total_time = time.perf_counter() - start_time

    # Save training examples
    save_file = "training_data_self_play.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(training_examples, f)

    print("\n=== Self-play timing summary ===")
    print(f"Games played:           {num_games}")
    print(f"Total time:             {total_time:.2f}s")
    print(f"Avg time per game:      {total_time / num_games:.2f}s")
    print(f"Total training samples: {len(training_examples)}")
    print(f"Samples per second:     {len(training_examples) / total_time:.2f}")

    print(f"\nSaved {len(training_examples)} training samples to {save_file}")

if __name__ == "__main__":
    main()
