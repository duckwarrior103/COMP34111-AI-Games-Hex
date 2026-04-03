import cProfile
import importlib
import pstats
import sys

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

SETUP_MOVES = [
    (Move(5, 5), Colour.RED),
    (Move(4, 4), Colour.BLUE),
]


def setup_board() -> Board:
    board = Board()
    for move, colour in SETUP_MOVES:
        board.set_tile_colour(move.x, move.y, colour)
    return board


def one_iteration_v0(mcts) -> None:
    leaf = mcts._select()
    child = mcts._expand(leaf) if not leaf.is_terminal else leaf
    reward = mcts._simulate(child)
    mcts._backpropagate(child, reward)


def one_iteration_v1(mcts) -> None:
    leaf = mcts._select()
    child = leaf.expand() if not leaf.is_terminal else leaf
    reward, moves = mcts._simulate(child)
    mcts._backpropagate(child, reward, moves)


def one_iteration_v2(mcts) -> None:
    leaf = mcts._select()
    child = leaf.expand() if not leaf.is_terminal else leaf
    reward, moves = mcts._simulate(child)
    mcts._backpropagate(child, reward, moves)


def one_iteration_v3(mcts) -> None:
    leaf = mcts._select()
    child = leaf.expand() if not leaf.is_terminal else leaf
    winner, moves = mcts._simulate(child)
    mcts._backpropagate(child, winner, moves)


def profile_version(label, mcts_module, mcts_class, iteration_fn, n_iters=100):
    """Set up the MCTS tree, then profile n_iters single iterations."""
    mod = importlib.import_module(mcts_module)
    cls = getattr(mod, mcts_class)

    mcts = cls(Colour.RED)
    mcts.update(setup_board(), SETUP_MOVES[-1][0])

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_iters):
        iteration_fn(mcts)
    profiler.disable()

    print(f"\n{'='*60}")
    print(f"  PROFILE: {label}  ({n_iters} iterations)")
    print(f"{'='*60}")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.strip_dirs()

    print(f"\n--- Top by cumulative time ---")
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    print(f"\n--- Top by total time ---")
    stats.sort_stats("tottime")
    stats.print_stats(20)


VERSIONS = [
    ("V0", "agents.Group21.v0.MCTSV0", "MCTSV0", "root",  one_iteration_v0),
    ("V1", "agents.Group21.v1.MCTSV1", "MCTSV1", "_root", one_iteration_v1),
    ("V2", "agents.Group21.v2.MCTSV2", "MCTSV2", "_root", one_iteration_v2),
    ("V3", "agents.Group21.v3.MCTSV3", "MCTSV3", "_root", one_iteration_v3),
]

N_ITERS = 200

if __name__ == "__main__":
    print(f"Profiling single MCTS iterations ({N_ITERS} per version)")

    for label, mod, cls, root_attr, iter_fn in VERSIONS:
        profile_version(label, mod, cls, iter_fn, n_iters=N_ITERS)