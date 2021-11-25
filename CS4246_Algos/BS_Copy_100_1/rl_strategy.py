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
Version date: 6/11/21

Included breached and scored locations
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
        self.enemy_locations = self.initialise_enemy_mobile()
        self.breached_locations = [0 for i in range(len(self.mobile_locations))]
        self.scored_locations = self.breached_locations.copy()

        self.done = False # Whether the game has ended
        self.updated_last = False # Whether the final update with terminal rewards is done
        self.previous_state = None
        
        self.previous_defence_states = []
        self.previous_defences = []
        self.previous_defence_next_states = []

        self.previous_attack_states = []
        self.previous_attacks = []
        self.previous_attack_next_states = []

        
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
        defence_agent_file = os.path.join(algo_path, "defence_model")
        attack_agent_file = os.path.join(algo_path, "attack_model")
        defence_memo_file = os.path.join(algo_path, "defence.npz")
        attack_memo_file = os.path.join(algo_path, "attack.npz")
        defence_epsilon_file = os.path.join(algo_path, "defence_epsilon.npz")
        attack_epsilon_file = os.path.join(algo_path, "attack_epsilon.npz")
        
        # Initialise agents
        self.defence_agent = Agent(fname=(defence_agent_file, defence_memo_file, defence_epsilon_file), 
                                   alpha=0.005, gamma=1, num_actions=len(self.defence_actions), 
                                   memory_size=10000, batch_size=64, epsilon_min=0.0, epsilon_dec=0.9998,
                                   input_shape=481, epsilon=1.0)
        self.attack_agent = Agent(fname=(attack_agent_file, attack_memo_file, attack_epsilon_file), 
                                  alpha=0.005, gamma=1, num_actions=len(self.attack_actions), 
                                  memory_size=10000, batch_size=64, epsilon_min=0.0, epsilon_dec=0.9998,
                                  input_shape=481, epsilon=1.0)

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine. 
        """
        game_state = gamelib.GameState(self.config, turn_state)
        game_state.enable_warnings = False
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        # game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        current_state = self.retrieve_state(game_state)
        
        # Check if there are previous experiences to learn from first
        if len(self.previous_defence_states) > 0 or len(self.previous_attack_states) > 0:
            # Calculate reward
            # reward = self.reward_function(self.previous_state, turn_state)
            attack_reward = self.reward_function(self.previous_state, turn_state, 0)
            defence_reward = self.reward_function(self.previous_state, turn_state, 0)

            # Remember the previous experiences. All previous actions will have the same reward
            # Remember for each agent
            for i in range(len(self.previous_defences)):
                self.defence_agent.remember(self.previous_defence_states[i], self.previous_defences[i], attack_reward, 
                                            self.previous_defence_next_states[i], self.done)
            
            for i in range(len(self.previous_attacks)):
                self.attack_agent.remember(self.previous_attack_states[i], self.previous_attacks[i], defence_reward, 
                                           self.previous_attack_next_states[i], self.done)
            
            # Agent learn from replay memory
            self.attack_agent.learn()
            self.defence_agent.learn()

            # Clear previous saved states to record new ones
            self.previous_defence_states.clear()
            self.previous_defences.clear()
            self.previous_defence_next_states.clear()
            self.previous_attack_states.clear()
            self.previous_attacks.clear()
            self.previous_attack_next_states.clear()
        
        next_s = current_state # Copy of current state for updating

        # Deploy structure units for maximum 10 actions or when run out of SB
        i = 0
        while i < 10 and game_state.get_resource(0) >= 2:
            self.previous_defence_states.append(np.copy(next_s))
            
            defence_action_index, defence_action = self.choose_and_execute_action(self.defence_agent, self.defence_actions, game_state, next_s)
            
            self.previous_defences.append(defence_action_index)
            
            if defence_action[0] == "END":
                self.previous_defence_next_states.append(np.copy(next_s))
                break

            next_s = self.retrieve_state(game_state)
            self.previous_defence_next_states.append(np.copy(next_s))
            i += 1
        
        # Deploy mobile units for maximum 10 actions or when run out of MB
        i = 0
        while i < 10 and game_state.get_resource(1) >= 1:
            self.previous_attack_states.append(np.copy(next_s))

            attack_action_index, attack_action = self.choose_and_execute_action(self.attack_agent, self.attack_actions, game_state, next_s)

            self.previous_attacks.append(attack_action_index)

            if attack_action[0] == "END":
                self.previous_attack_next_states.append(np.copy(next_s))
                break
            
            next_s = self.retrieve_state(game_state)
            self.previous_attack_next_states.append(np.copy(next_s))
            i += 1

        # Save previous turn state to compute reward
        self.previous_state = turn_state
        # Reset breached and scored locations
        self.reset_breached_and_scored()
        game_state.submit_turn()

    def choose_and_execute_action(self, agent, action_space, game_state, state):
        """
        This function makes agent choose and execute a valid action.
        """
        
        random = agent.decide_random_or_not()
        gamelib.debug_write(f"epsilon = {agent.epsilon}")
        # if random is true, keep generating random action until getting a valid action
        if random:
            while True:
                action_index = agent.choose_action(state, random)
                action = action_space[action_index]
                if action[0] == "END":
                    return action_index, action
                if self.execute_action(action, game_state):
                    return action_index, action
        # if random is false, find the first valid action with the best Q_value
        else:
            action_indices = agent.choose_action(state, random)
            for i in range(action_indices.shape[0]):
                action_index = action_indices[i]
                action = action_space[action_index]
                if action[0] == "END":
                    return action_index, action
                if self.execute_action(action, game_state):
                    return action_index, action

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Record breached locations
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0] # Location is a list [x, y] or [y, x]
            damage = breach[1]
            # for location on board = 0. If there is damage, 0 += damage
            # Check owner of the mobile unit
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self: # If the unit belongs to the opponent, means they attacked me
                # Get index of breached location
                breach_index = self.mobile_locations.index(location)
                self.breached_locations[breach_index] += damage
                gamelib.debug_write("Got scored on at: {}".format(location))
            else: # Unit belongs to me, means I attacked them
                # Get index of attacked location
                attack_index = self.enemy_locations.index(location)
                self.scored_locations[attack_index] += damage
                gamelib.debug_write("Scored on at: {}".format(location))

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
            self.defence_agent.remember(self.previous_defence_states[i], self.previous_defences[i], reward, 
                                        self.previous_defence_next_states[i], self.done)
        for i in range(len(self.previous_attacks)):
            self.attack_agent.remember(self.previous_attack_states[i], self.previous_attacks[i], reward, 
                                        self.previous_attack_next_states[i], self.done)
        
        # Agent learn from replay memory
        self.attack_agent.learn()
        self.defence_agent.learn()

        # Agent save model and replay memory
        if not os.path.exists('.test'):
            self.attack_agent.save_model_and_memory()
            self.defence_agent.save_model_and_memory()

        # save results to file
        win = 1 if winner == 1 else 0
        turns = state['endStats']['turns']
        health = state['p1Stats'][0]
        opp_health = state['p2Stats'][0]
        with open(os.path.join(os.path.dirname(__file__), 'results.csv'), 'a+') as f:
            f.write(f'{win}, {turns}, {health}, {opp_health}\n')

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

    def initialise_enemy_mobile(self):
        """
        Initialise locations where enemy mobile units can be deployed
        AKA the locations where our units will scoree
        Size of list: 28
        """
        y = 14
        x = 0
        enemy_locations = []
        while x < 14:
            enemy_locations.append([x     , y])
            enemy_locations.append([27 - x, y])
            x += 1
            y += 1
        return enemy_locations

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
        success = False
        if unit == UPGRADE:
            if game_state.attempt_upgrade(coordinates) == 1:
              success = True
        else:
            if game_state.attempt_spawn(unit, coordinates) == 1:
              success = True
        return success

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
        - Breached locations
        - Scored locations
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
        # Add breached and scored locations
        state.extend(self.breached_locations)
        state.extend(self.scored_locations)

        return np.array(state, dtype=np.float16)

    def reset_breached_and_scored(self):
        """
        Resets the breached and scored locations, so that on_action_frame can be updated
        """
        self.breached_locations = [0 for i in range(len(self.mobile_locations))]
        self.scored_locations = self.breached_locations.copy()

    def attack_reward(self, state):
        state = json.loads(state)
        return 30 - state['p2Stats'][0]

    def defence_reward(self, state):
        state = json.loads(state)
        return state['p1Stats'][0]

    def count_structure_points(self, units, map):
        total_structure = len(units[UNIT_TYPE_TO_INDEX[WALL]]) + \
                            len(units[UNIT_TYPE_TO_INDEX[SUPPORT]]) * 4 + \
                            len(units[UNIT_TYPE_TO_INDEX[TURRET]]) * 2

        upgrades = units[UNIT_TYPE_TO_INDEX[UPGRADE]]
        for unit in upgrades:
            unit_type = map[unit[0], unit[1]][0].unit_type
            if unit_type == WALL:
                total_structure += 1
            elif unit_type == SUPPORT or unit_type == TURRET:
                total_structure += 4

        return total_structure

    def state_reward(self, state, health=10, structure=1):
        """
        Reward for a particular state based on player and opponent stats and 
        usefulness of the deployed mobile and structure units.
        """
        game_state = gamelib.GameState(self.config, state)
        json_state = json.loads(state)

        p1_health = json_state['p1Stats'][0]
        p2_health = json_state['p2Stats'][0]

        # reward based on unit cost
        p1_total_structure = json_state['p1Stats'][1] + \
                self.count_structure_points(json_state['p1Units'], game_state.game_map)
        p2_total_structure = json_state['p2Stats'][1] + \
                self.count_structure_points(json_state['p2Units'], game_state.game_map)

        return health * (p1_health - p2_health) + \
                structure * (p1_total_structure - p2_total_structure)

    def reward_function(self, prev_state, state, t=0):
        """
        Reward difference between previous and current state
        """
        if prev_state is None:
            return 0

        reward_type = [
            self.state_reward, 
            self.attack_reward, 
            self.defence_reward
        ]

        return reward_type[t](state) - reward_type[t](prev_state)


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
