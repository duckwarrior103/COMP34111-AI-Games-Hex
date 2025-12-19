import json
import math
import os
import subprocess
import argparse
import time
from multiprocessing import Pool, cpu_count

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
UP = "\u2191"
DOWN = "\u2193"

GAME_STATS_PATH  = 'game_stats.json'

def load_stats():
    if os.path.exists(GAME_STATS_PATH):
        with open(GAME_STATS_PATH, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_stats(stats: dict):
    with open(GAME_STATS_PATH, 'w') as file:
        json.dump(stats, file, indent=4)

def update_pair_stats(p1: str, p2: str, games_played: int, win_rate: float, avg_game_time: float):
    key = f'{p1}-{p2}'
    stats = load_stats()
    if key not in stats:
        stats[key] = {
            'games': 0,
            'avg_win_rate': 0.0,
            'avg_game_time': 0.0
        }

    pair_stats = stats[key]
    old_games = pair_stats['games']
    pair_stats['games'] += games_played
    pair_stats['avg_win_rate'] = (pair_stats['avg_win_rate'] * old_games + win_rate * games_played) / pair_stats['games']
    pair_stats['avg_game_time'] = (pair_stats['avg_game_time'] * old_games + avg_game_time * games_played) / pair_stats['games']

    save_stats(stats)

def get_average_statistics(p1: str, p2: str) -> tuple[float, float] | None:
    key = f'{p1}-{p2}'
    stats = load_stats()
    if key not in stats:
        return None

    return stats[key]['avg_win_rate'], stats[key]['avg_game_time']

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
    parser.add_argument("-p2", "--player2", type=str, default="agents.Group21.v3.MCTSAgentV3 MCTSAgentV3")
    # parser.add_argument("-p2", "--player2", type=str, default="agents.Group21.RandomAgent RandomAgent")

    args = parser.parse_args()

    # Extract the class name and use as player names
    p1_name = args.player1.split(" ")[1]
    p2_name = args.player2.split(" ")[1]

    # If the two names are the same, then append a 2 to the second player
    if p1_name == p2_name:
        p2_name += '2'

    start_time = time.time()
    wins, total, game_times = run_games(args.num_games, args.player1, p1_name, args.player2, p2_name)
    time_taken = time.time() - start_time

    # TODO: Fix this ugly ass code if we have time
    win_rate = wins / total
    avg_game_time = sum(game_times) / len(game_times)
    old_avg_wr, old_avg_game_time = get_average_statistics(p1_name, p2_name) or (win_rate, avg_game_time)
    diff_wr = abs(old_avg_wr - win_rate)
    diff_game_time = abs(old_avg_game_time - avg_game_time)

    print("\n=== RESULTS ===")
    print(f"Total Games: {total}")
    print(f"Wins for {p1_name} against {p2_name}: {wins}")
    print(f"Win Rate: {win_rate*100:.1f}% {GREEN if win_rate >= old_avg_wr else RED}{UP if win_rate >= old_avg_wr else DOWN} {diff_wr*100:.1f}%{RESET}")

    print("\n=== TIME STATISTICS ===")
    print(f"Average Game Time: {avg_game_time:.1f} s {GREEN if avg_game_time <= old_avg_game_time else RED}{DOWN if avg_game_time <= old_avg_game_time else UP} {diff_game_time:.1f}s{RESET}")
    print(f"Fastest Game: {min(game_times):.1f}s")
    print(f"Slowest Game: {max(game_times):.1f}s")
    print(f"Total Time Taken: {time_taken:.1f}s")

    update_pair_stats(p1_name, p2_name, total, win_rate, avg_game_time)
