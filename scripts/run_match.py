import os
import subprocess
import sys

# Runs a single game
def run_single_game(process_command):
    log = open('match.log', 'a+')
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=log,
        stderr=log
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()


def results_update(results_file, update, current):
    win_count = 0
    turns_count = 0
    total_health = 0
    total_opp_health = 0

    with open(results_file, 'r') as f:
        results = f.readlines()[-update:]
        for result in results:
            values = [float(x) for x in result.split(', ')]
            win_count += values[0]
            turns_count += values[1]
            total_health += values[2]
            total_opp_health += values[3]

    print(f'[Matches {current-update+1}-{current}] '
          f'Total wins: {int(win_count)}/{update}, '
          f'Avg turns: {int(turns_count)/update}, '
          f'Avg health: {total_health/update}, '
          f'Avg opp health: {total_opp_health/update}')


def main():
    # Get location of this run file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(file_dir, os.pardir)
    parent_dir = os.path.abspath(parent_dir)

    # Get if running in windows OS
    is_windows = sys.platform.startswith('win')
    print("Is windows: {}".format(is_windows))

    # Set default path for algos if script is run with no params
    default_algo = parent_dir + "\\python-algo\\run.ps1" if is_windows else parent_dir + "/python-algo/run.sh"
    algo1 = default_algo
    algo2 = default_algo
    num_plays = 1
    update_episode = 1

    # If script run with params, use those algo locations when running the game
    if len(sys.argv) > 1:
        algo1 = sys.argv[1]
    if len(sys.argv) > 2:
        algo2 = sys.argv[2]
    if len(sys.argv) > 3:
        num_plays = int(sys.argv[3])
    if len(sys.argv) > 4:
        update_episode = int(sys.argv[4])

    # If folder path is given instead of run file path, add the run file to the path based on OS
    # trailing_char deals with if there is a trailing \ or / or not after the directory name
    if is_windows:
        if "run.ps1" not in algo1:
            trailing_char = "" if algo1.endswith("\\") else "\\"
            algo1 = algo1 + trailing_char + "run.ps1"
        if "run.ps1" not in algo2:
            trailing_char = "" if algo2.endswith("\\") else "\\"
            algo2 = algo2 + trailing_char + "run.ps1"
    else:
        if "run.sh" not in algo1:
            trailing_char = "" if algo1.endswith('/') else "/"
            algo1 = algo1 + trailing_char + "run.sh"
        if "run.sh" not in algo2:
            trailing_char = "" if algo2.endswith('/') else "/"
            algo2 = algo2 + trailing_char + "run.sh"

    print("Algo 1: ", algo1)
    print("Algo 2:", algo2)

    # delete log file and results before running
    if os.path.exists('match.log'):
        os.remove('match.log')
    results_file = os.path.join(os.path.dirname(algo1),'results.csv')
    if os.path.exists(results_file):
        os.remove(results_file)

    # run game
    print()
    for i in range(1, num_plays+1):
        print(f"Running match {i}", end='\r')
        run_single_game("cd {} && java -jar engine.jar work {} {}".format(parent_dir, algo1, algo2))
        if i % update_episode == 0:
            results_update(results_file, update_episode, i)
        elif i == num_plays:
            results_update(results_file, num_plays % update_episode, i)
    
    print('\nDone!')

if __name__ == "__main__":
    main()