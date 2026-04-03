import copy
import importlib
import statistics

from agents.Group21.v0.MCTSV0 import MCTSV0
from agents.Group21.v1.MCTSV1 import MCTSV1
from agents.Group21.v2.MCTSV2 import MCTSV2
from agents.Group21.v3.MCTSV3 import MCTSV3
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Pre-place 2 stones so V3 skips its opening book and runs MCTS
SETUP_MOVES = [
    (Move(5, 5), Colour.RED),
    (Move(4, 4), Colour.BLUE),
]

PREV_MOVE = Move(4, 4)


def setup_board() -> Board:
    board = Board()
    for move, colour in SETUP_MOVES:
        board.set_tile_colour(move.x, move.y, colour)
    return board


def bench_v0() -> int:
    mcts = MCTSV0(Colour.RED)
    mcts.update(setup_board(), PREV_MOVE)
    root = mcts.root
    mcts.run()
    return root.N


def bench_v1() -> int:
    mcts = MCTSV1(Colour.RED)
    mcts.update(setup_board(), Move(4, 4))
    root = mcts._root
    mcts.run()
    return root.N


def bench_v2() -> int:
    mcts = MCTSV2(Colour.RED)
    mcts.update(setup_board(), Move(4, 4))
    root = mcts._root
    mcts.run()
    return root.N


def bench_v3() -> int:
    mcts = MCTSV3(Colour.RED)
    mcts.update(setup_board(), Move(4, 4))
    root = mcts._root
    mcts.run()
    return root.N

BENCHMARKS = [
    ("V0", bench_v0),
    ("V1", bench_v1),
    ("V2", bench_v2),
    ("V3", bench_v3),
]

NUM_TRIALS = 5

def main():
    print(f"MCTS Simulation Throughput Benchmark")
    print(f"Trials per version: {NUM_TRIALS}")
    print(f"{'='*60}")

    results = {}

    for label, bench_fn in BENCHMARKS:
        counts = []
        for i in range(NUM_TRIALS):
            n = bench_fn()
            counts.append(n)
            print(f"  {label} trial {i+1}: {n:,} simulations")

        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0
        results[label] = (mean, stdev, min(counts), max(counts))
        print(f"  => mean: {mean:,.0f}  stdev: {stdev:,.0f}  "
              f"range: [{min(counts):,}, {max(counts):,}]")
        print()

    # Summary table
    print(f"{'='*60}")
    print(f"{'Version':<30} {'Mean':>8} {'Stdev':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, (mean, stdev, mn, mx) in results.items():
        print(f"{label:<30} {mean:>8,.0f} {stdev:>8,.0f} {mn:>8,} {mx:>8,}")


if __name__ == "__main__":
    main()
