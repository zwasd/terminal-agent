import gamelib
import random
import math
import warnings
from sys import maxsize
import numpy as np
import json
import tensorflow as tf
from rl_agent import Agent # Import RL Agent
from sklearn.preprocessing import OneHotEncoder

"""
Added functions:
- initialise_mobile(): generates all coordinates that can spawn a mobile unit
- initialise_structure(): generates all coordinates that can spawn a structure
- generate_actions(): generate all actions an agent can take
- execute_actions(): execute an action based on output of DQN (to be refined)
- retrieve_state(): retrieves information about current state

To be done:
- Settle how the agent will take in the state and process actions
- Reward function
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self, agent, load=False):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        # Additions
        self.mobile_locations = self.initialise_mobile()
        self.structure_locations = self.initialise_structure()
        self.actions = self.generate_actions()
        self.agent = agent
        if load:
            agent.load_model("rl_agent.h5") # For loading in old models

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1 # Mobile points
        SP = 0 # Structure points

        # This is a good place to do initial setup
        self.scored_on_locations = []

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
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        # Put our agent action here
        """
        While turn not over:
            agent chooses action
            Get next state, reward, and done (whether the turn is over)
            Remember the experience in replay buffer
            Update to next state
            Let the agent learn from the action
        """

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
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))

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
            mobile_locations.append((x     , y))
            mobile_locations.append((27 - x, y))
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
                structure_locations.append((x     , 13 - _y))
                structure_locations.append((27 - x, 13 - _y))
            y += 1
        return structure_locations

    def generate_actions(self):
        """
        Generate all possible actions that the agent can take
        Each action is a tuple: (unit, coordinate)

        If we allow for upgrades, the action space is 28 + 210 + 210...
        """
        actions = []
        for ml in self.mobile_locations:
            # Mobile units: SCOUT, DEMOLISHER, INTERCEPTOR
            actions.append((SCOUT, ml))
            actions.append((DEMOLISHER, ml))
            actions.append((INTERCEPTOR, ml))
        for sl in self.structure_locations:
            # Structure units: WALL, SUPPORT, TURRET
            actions.append((WALL, sl))
            actions.append((SUPPORT, sl))
            actions.append((TURRET, sl))
            # Allow for upgrades - I think this function does not need a specific unit type
            # But will have to check for the string, to know whether to deploy or upgrade unit
            actions.append(("UPGRADE", sl))
        # Add action to end turn
        actions.append(("END", 0))
        
        return np.array(actions)

    def execute_action(self, action, game_state):
        """
        Execute action based on the chosen action from DQN
        - action: tuple of unit and coordinate, same as from generate_actions
        - game_state: the current game_state
        Checks unit type and executes action at corresponding coordinate
        """
        unit, coordinates = action
        if unit == "UPGRADE":
            game_state.attempt_upgrade(coordinates)
        elif unit == "END":
            game_state.submit_turn()
        else:
            game_state.attempt_spawn(unit, coordinates)

    def retrieve_state(self, game_state):
        """
        Retrieve all information about the current state to feed into DQN
        Takes in a game_state
        Returns a np array containing:
        - My health
        - Enemy health
        - My MP
        - My SP
        - Every valid position on the board, with its corresponding unit, if there is one
        """
        state = []
        # Add my health, opponent health, my MP and SP
        state.append(game_state.my_health)
        state.append(game_state.enemy_health)
        state.append(MP)
        state.append(SP)
        # Convert units to one hot encoding
        units = np.asarray[WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR]
        encoder = OneHotEncoder(sparse=False)
        one_hot_units = encoder.fit_transform(units)
        units_dict = {unit: encode for unit, encode in zip(units, one_hot_units)}
        # Get state of every position on the board
        for y in range(28):
            for x in range(28):
                # Check if is valid position:
                if game_state.game_map.in_arena_bounds((x, y)):
                    # If no unit at the position, add list of zeros
                    if len(game_state.game_map[x, y]) == 0:
                        state.append([0, 0, 0, 0, 0, 0])
                    # Else, add the corresponding unit
                    else:
                        state.append(units_dict[game_state.game_map[x, y][0]])
        
        return np.array(state)


    def import_Q(self):
        """
        This function will allow us to import a Q-table from previous training
        sessions to allow for discontinuous training, or uploading
        """
        pass

if __name__ == "__main__":
    # Initialise agent
    # rl_agent = Agent(alpha=0.005, gamma=1, num_actions=len(self.actions), 
    #                  memory_size=1000000, batch_size=64, epsilon_min=0.01)
    algo = AlgoStrategy(rl_agent)
    algo.start()
