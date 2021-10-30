import gamelib
import random
import math
import warnings
from sys import maxsize
import numpy as np
import json
import tensorflow as tf
from rl_agent import Agent # Import RL Agent
import os

"""
Version date: 26/10/21

- Shifted agent and action initialisation from __init__ to on_game_start
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        # gamelib.debug_write('Random seed: {}'.format(seed))

        # Additions
        self.mobile_locations = self.initialise_mobile()
        self.structure_locations = self.initialise_structure()
        # generate_actions() cannot be done here as the global parameters are not initialised yet
        # We can generate_actions on_game_start
        # self.defence_actions = [] 
        # self.attack_actions = []
        self.done = False # Whether the game has ended
        self.updated_last = False # Whether the final update with terminal rewards is done
        self.previous_state = None
        self.previous_states = []
        self.previous_defences = []
        self.previous_attacks = []
        self.previous_next_states = []

        

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        # gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, REMOVE, UPGRADE, MP, SP, UNIT_TYPE_TO_INDEX

        UNIT_TYPE_TO_INDEX = {}
        
        WALL = config["unitInformation"][0]["shorthand"]
        UNIT_TYPE_TO_INDEX[WALL] = 0
        
        SUPPORT = config["unitInformation"][1]["shorthand"]
        UNIT_TYPE_TO_INDEX[SUPPORT] = 1
        
        TURRET = config["unitInformation"][2]["shorthand"]
        UNIT_TYPE_TO_INDEX[TURRET] = 2
        
        SCOUT = config["unitInformation"][3]["shorthand"]
        UNIT_TYPE_TO_INDEX[SCOUT] = 3
        
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        UNIT_TYPE_TO_INDEX[DEMOLISHER] = 4
        
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        UNIT_TYPE_TO_INDEX[INTERCEPTOR] = 5
        
        REMOVE = config["unitInformation"][6]["shorthand"]
        UNIT_TYPE_TO_INDEX[REMOVE] = 6
        
        UPGRADE = config["unitInformation"][7]["shorthand"]
        UNIT_TYPE_TO_INDEX[UPGRADE] = 7
        
        MP = 1 # Mobile points
        SP = 0 # Structure points

        # This is a good place to do initial setup
        self.game_map = gamelib.GameMap(config)
        self.scored_on_locations = []
        self.defence_actions, self.attack_actions = self.generate_actions()

        algo_path = os.path.dirname(__file__)
        defence_agent_file = os.path.join(algo_path, "defence.h5")
        attack_agent_file = os.path.join(algo_path, "attack.h5")
        defence_memo_file = os.path.join(algo_path, "defence.npz")
        attack_memo_file = os.path.join(algo_path, "attack.npz")
        defence_epsilon_file = os.path.join(algo_path, "defence_epsilon.npz")
        attack_epsilon_file = os.path.join(algo_path, "attack_epsilon.npz")
        
        # Initialise agents
        self.defence_agent = Agent(fname=(defence_agent_file, defence_memo_file, defence_epsilon_file), 
                                   alpha=0.005, gamma=1, num_actions=len(self.defence_actions), 
                                   memory_size=10000, batch_size=64, epsilon_min=0.01, 
                                   input_shape=425, epsilon=0.5)
        self.attack_agent = Agent(fname=(attack_agent_file, attack_memo_file, attack_epsilon_file), 
                                  alpha=0.005, gamma=1, num_actions=len(self.attack_actions), 
                                  memory_size=10000, batch_size=64, epsilon_min=0.01, 
                                  input_shape=425, epsilon=0.5)


    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        # game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        """ Put our agent action here """
        current_state = self.retrieve_state(game_state)
        
        # Check if there are previous experiences to learn from first
        if len(self.previous_states) > 0:
            # Calculate reward
            reward = self.reward_function(self.previous_state, turn_state)

            # Remember the previous experiences. All previous actions will have the same reward
            # Remember for each agent
            for i in range(len(self.previous_defences)):
                self.defence_agent.remember(self.previous_states[i], self.previous_defences[i], reward, 
                                            self.previous_next_states[i], self.done)
            for i in range(len(self.previous_attacks)):
                self.attack_agent.remember(self.previous_states[i], self.previous_attacks[i], reward, 
                                           self.previous_next_states[i], self.done)
            
            # Agent learn from replay memory
            self.attack_agent.learn()
            self.defence_agent.learn()

            # Clear previous saved states to record new ones
            self.previous_states.clear()
            self.previous_attacks.clear()
            self.previous_defences.clear()
            self.previous_next_states.clear()
        
        next_s = current_state # Copy of current state for updating
        defence_end = False
        attack_end = False

        # For a maximum of 10 actions
        for i in range(10):
            if defence_end and attack_end:
                break

            # Add next_s to previous_states
            self.previous_states.append(next_s)
            
            # Attack agent choose action
            if not attack_end:
                attack_action_index = self.attack_agent.choose_action(next_s)
                self.previous_attacks.append(attack_action_index)
                attack_action = self.attack_actions[attack_action_index]
                # If agent chooses to END
                if attack_action[0] == "END":
                    attack_end = True
                # Otherwise, execute action
                else:
                    self.execute_action(attack_action, game_state)

            # Defence agent choose action
            if not defence_end:
                defence_action_index = self.defence_agent.choose_action(next_s)
                self.previous_defences.append(defence_action_index)
                defence_action = self.defence_actions[defence_action_index]
                # If agent chooses to END
                if defence_action[0] == "END":
                    defence_end = True
                # Otherwise, execute action
                else:
                    self.execute_action(defence_action, game_state)

            # Update state after deploying units and add to previous_next_states
            next_s = self.retrieve_state(game_state)
            self.previous_next_states.append(next_s)

        self.previous_state = turn_state
        game_state.submit_turn()

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                # gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                # gamelib.debug_write("All locations: {}".format(self.scored_on_locations))

    def on_game_end(self, game_state):
        # We should be able to use the state json string
        state = json.loads(game_state)
        self.done = True
        # Winner: 1 is ourselves, 2 is the opponent
        winner = state['endStats']['winner']
        reward = 10000 if winner == 1 else -10000

        # Remember the previous experiences. All previous actions will have the same reward
        # Remember for each agent
        for i in range(len(self.previous_defences)):
            self.defence_agent.remember(self.previous_states[i], self.previous_defences[i], reward, 
                                        self.previous_next_states[i], self.done)
        for i in range(len(self.previous_attacks)):
            self.attack_agent.remember(self.previous_states[i], self.previous_attacks[i], reward, 
                                        self.previous_next_states[i], self.done)
        
        # Agent learn from replay memory
        self.attack_agent.learn()
        self.defence_agent.learn()

        # Agent save model and replay memory
        self.attack_agent.save_model_and_memory()
        self.defence_agent.save_model_and_memory()

    def initialise_mobile(self):
        """
        Initialise locations where mobile units can be deployed
        Returns a list of locations
        Size of list: 28
        """
        y = 13
        x = 0
        mobile_locations = []
        while x < 14:
            mobile_locations.append([x     , y])
            mobile_locations.append([27 - x, y])
            x += 1
            y -= 1
        return mobile_locations

    def initialise_structure(self):
        """
        Initialise locations where structures can be deployed
        Returns a list of locations
        Size of list: 210
        """
        y = 1
        structure_locations = []
        for x in range(14):
            for _y in range(y):
                structure_locations.append([x     , 13 - _y])
                structure_locations.append([27 - x, 13 - _y])
            y += 1
        return structure_locations

    def generate_actions(self):
        """
        Generate all possible actions that the agent can take
        Each action is a tuple: (unit, coordinate)

        If we allow for upgrades, the action space is 28 + 210 + 210...
        """
        defence_actions = []
        attack_actions = []
        for ml in self.mobile_locations: # FOR ATTACK
            # Mobile units: SCOUT, DEMOLISHER, INTERCEPTOR
            attack_actions.append((SCOUT, ml))
            attack_actions.append((DEMOLISHER, ml))
            attack_actions.append((INTERCEPTOR, ml))
        for sl in self.structure_locations: # FOR DEFENCE
            # Structure units: WALL, SUPPORT, TURRET
            defence_actions.append((WALL, sl))
            defence_actions.append((SUPPORT, sl))
            defence_actions.append((TURRET, sl))
            # Allow for upgrades - I think this function does not need a specific unit type
            # But will have to check for the string, to know whether to deploy or upgrade unit
            defence_actions.append((UPGRADE, sl))
        # Add action to end turn
        defence_actions.append(("END", 0))
        attack_actions.append(("END", 0))
        
        return np.array(defence_actions), np.array(attack_actions)

    def execute_action(self, action, game_state):
        """
        Execute action based on the chosen action from DQN
        - action: tuple of unit and coordinate, same as from generate_actions
        - game_state: the current game_state
        Checks unit type and executes action at corresponding coordinate
        End handling will be done outside, in on_turn function
        """
        unit, coordinates = action
        if unit == UPGRADE:
            game_state.attempt_upgrade(coordinates)
        else:
            game_state.attempt_spawn(unit, coordinates)

    def retrieve_state(self, game_state):
        """
        Retrieve all information about the current state to feed into DQN
        Takes in a game_state
        Returns a np array containing:
        - My health
        - Enemy health
        - Turn count
        - My MP
        - My SP
        - Every valid position on the board, with its corresponding unit, if there is one
        Return state here is 1-dimensional
        """
        state = []
        # Add my health, opponent health, my MP and SP
        state.append(game_state.my_health)
        state.append(game_state.enemy_health)
        state.append(game_state.turn_number)
        state.append(MP)
        state.append(SP)
        # Get state of every position on the board
        for y in range(28):
            for x in range(28):
                # Check if is valid position:
                if game_state.game_map.in_arena_bounds((x, y)):
                    # If no unit at the position, add minus one
                    if len(game_state.game_map[x, y]) == 0:
                        state.append(-1)
                    # Else, add the corresponding unit
                    else:
                        state.append(UNIT_TYPE_TO_INDEX[game_state.game_map[x, y][0].unit_type])

        return np.array(state, dtype=np.float16)

    # Basic reward functions for testing
    # def attack_reward(self, previous_state, current_state):
    #     """
    #     Reward calculation for attack agent
    #     Reward = difference between the previous enemy health and current enemy health
    #     Does not account for terminal rewards; those are handled directly in 
    #     """
    #     previous_enemy_health = previous_state[1]
    #     current_enemy_health = current_state[1]
    #     reward = -0.04
    #     # If damage was done to enemy
    #     if current_enemy_health < previous_enemy_health:
    #         reward = previous_enemy_health - current_enemy_health
    #     return reward

    # def defence_reward(self, previous_state, current_state):
    #     previous_my_health = previous_state[0]
    #     current_my_health = current_state[0]
    #     reward = -0.04
    #     # If damage was done to agent
    #     if current_my_health < previous_my_health:
    #         reward = current_my_health - previous_my_health
    #     return reward

    def attack_reward(self, state):
        pass

    def defence_reward(self, state):
        pass

    def state_reward(self, state, terminal, health, structure):
        """
        Reward for a particular state based on player and opponent stats and 
        usefulness of the deployed mobile and structure units.
        """
        state = json.loads(state)

        # game end
        if state['turnInfo'] == 2:
            return terminal / math.log(state['endStats']['turns'] + 1) * \
                    -1 if state['endStats']['winner'] == 2 else 1

        p1_health = state['p1Stats'][0]
        p2_health = state['p2Stats'][0]

        # reward based on unit cost
        p1_total_structure = 0
        p2_total_structure = 0

        p1_units = state['p1Units']
        p2_units = state['p2Units']


        p1_total_structure += len(p1_units[UNIT_TYPE_TO_INDEX[WALL]]) + \
                                len(p1_units[UNIT_TYPE_TO_INDEX[SUPPORT]]) * 4 + \
                                len(p1_units[UNIT_TYPE_TO_INDEX[TURRET]]) * 2 + \
                                state['p1Stats'][1]
        p2_total_structure += len(p2_units[UNIT_TYPE_TO_INDEX[WALL]]) + \
                                len(p2_units[UNIT_TYPE_TO_INDEX[SUPPORT]]) * 4 + \
                                len(p2_units[UNIT_TYPE_TO_INDEX[TURRET]]) * 2 + \
                                state['p2Stats'][1]

        return health * (p1_health - p2_health) + \
                structure * (p1_total_structure - p2_total_structure)

    def reward_function(self, prev_state, state, terminal=10000000, health=10, structure=1):
        """
        Reward difference between previous and current state
        """
        if prev_state is None:
            return 0

        return self.state_reward(state, terminal, health, structure) - \
            self.state_reward(prev_state, terminal, health, structure)

            

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
