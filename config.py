# Environment
GRID_SIZE = 5
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2), (3, 1)]
MAX_STEPS = 50

# Training
EPISODES = 400
RUNS = 3

# Learning Parameters
ALPHA = 0.005
GAMMA = 0.99

# Exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.05