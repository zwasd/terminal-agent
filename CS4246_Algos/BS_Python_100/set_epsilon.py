import numpy as np
import sys
import os


algo_path = algo_path = os.path.dirname(__file__)
defence_epsilon_file = os.path.join(algo_path, "defence_epsilon.npz")
attack_epsilon_file = os.path.join(algo_path, "attack_epsilon.npz")

np.savez_compressed(defence_epsilon_file, epsilon = float(sys.argv[1]))
np.savez_compressed(attack_epsilon_file, epsilon = float(sys.argv[1]))