import subprocess
import argparse
import time
from multiprocessing import Pool, cpu_count


def extract_winner(output: str) -> str | None:
    lines = output.splitlines()
    for line in reversed(lines):
        if line.startswith("winner,"):
            return line.split(",")[1]
    return None

def run_single_game(args) -> tuple[str, float]:
    player1, player1_name, player2, player2_name = args

    start_time = time.time()
    result = subprocess.run(
        [
        "python3", "Hex.py",
        "-p1", player1,
        "-p1Name", player1_name,
        "-p2", player2,
        "-p2Name", player2_name
        ],
        capture_output=True,
        text=True
    )

    end_time = time.time()
    duration = end_time - start_time

    output = result.stdout + result.stderr
    winner = extract_winner(output)

    return winner, duration

def run_games(num_games: int, player1: str, player1_name: str, player2: str, player2_name: str) -> tuple[int, int, list[float]]:
    tasks = []
    for i in range(num_games):
        tasks.append((player1, player1_name, player2, player2_name))

    num_processes = min(num_games, cpu_count())

    total_wins = 0
    total_games = 0
    game_times = []

    print(f'Running {num_games} games using {num_processes} parallel processes...')
    with Pool(processes=num_processes) as pool:
        for winner, duration in pool.imap_unordered(run_single_game, tasks):
            if winner == player1_name:
                total_wins += 1
            total_games += 1
            game_times.append(duration)
            print(f'[{total_games}/{num_games}] Current WR: {total_wins / total_games * 100:.1f}%')

    return total_wins, total_games, game_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_games", type=int, default=10)
    parser.add_argument("-p1", "--player1", type=str, default="agents.Group21.MCTSAgent MCTSAgent")
    parser.add_argument("-p1Name", "--player1Name", type=str, default="Current MCTSAgent")
    parser.add_argument("-p2", "--player2", type=str, default="agents.Group21.old.MCTSAgentFri MCTSAgentFri")
    # parser.add_argument("-p2", "--player2", type=str, default="agents.Group21.RandomAgent RandomAgent")
    parser.add_argument("-p2Name", "--player2Name", type=str, default="Old MCTSAgent")

    args = parser.parse_args()

    start_time = time.time()
    wins, total, game_times = run_games(args.num_games, args.player1, args.player1Name, args.player2, args.player2Name)
    time_taken = time.time() - start_time

    print("\n=== RESULTS ===")
    print(f"Total Games: {total}")
    print(f"Wins for {args.player1Name} against {args.player2Name}: {wins}")
    print(f"Win Rate: {wins/total*100:.1f}%")

    print("\n=== TIME STATISTICS ===")
    print(f"Average Game Time: {sum(game_times)/len(game_times):.3f} s")
    print(f"Fastest Game: {min(game_times):.3f} s")
    print(f"Slowest Game: {max(game_times):.3f} s")
    print(f"Total Time Taken: {time_taken:.3f} s")
