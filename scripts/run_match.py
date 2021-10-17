import os
import subprocess
import sys

# Runs a single game
def run_single_game(process_command):
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()

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
num_plays = None

# If script run with params, use those algo locations when running the game
if len(sys.argv) > 1:
    algo1 = sys.argv[1]
if len(sys.argv) > 2:
    algo2 = sys.argv[2]
if len(sys.argv) > 3:
    num_plays = int(sys.argv[3])

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


if num_plays is not None:
  for i in range(1, num_plays+1):
    print(f"\nStart run match {i}")
    run_single_game("cd {} && java -jar engine.jar work {} {}".format(parent_dir, algo1, algo2))
    print(f"Finished running match {i}\n")
else:
    print("\nStart run a match")
    run_single_game("cd {} && java -jar engine.jar work {} {}".format(parent_dir, algo1, algo2))
    print("Finished running match\n")
