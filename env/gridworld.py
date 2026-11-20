import os 
class GridWorld:
    def __init__(self, size, start, goal, obstacles=None, max_steps=50):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []
        self.max_steps = max_steps

        self.action_space = ["up", "down", "left", "right"]

        self.reset()

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        x, y = self.state

        # Move logic
        if action == "up":
            x -= 1
        elif action == "down":
            x += 1
        elif action == "left":
            y -= 1
        elif action == "right":
            y += 1

        # Stay inside grid
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        next_state = (x, y)

        # Obstacle handling (bounce back)
        if next_state in self.obstacles:
            next_state = self.state

        self.state = next_state
        self.steps += 1

        # Rewards
        if self.state == self.goal:
            return self.state, 10, True

        if self.steps >= self.max_steps:
            return self.state, -1, True

        return self.state, -0.1, False

    def render(self, episode=None, step=None, agent=None):

       os.system("cls" if os.name == "nt" else "clear")

       print("=" * 45)

       if episode is not None:
           print(f"Episode : {episode}")

       if step is not None:
           print(f"Step    : {step}")

       if agent is not None:

           alpha = getattr(agent, "alpha", "N/A")
           gamma = getattr(agent, "gamma", "N/A")
           epsilon = getattr(agent, "epsilon", "N/A")

           print(
            f"Alpha={alpha} | "
            f"Gamma={gamma} | "
            f"Epsilon={epsilon:.3f}"
            )

       print("=" * 45)

       grid = [["." for _ in range(self.size)] for _ in range(self.size)]

       # obstacles
       for (x, y) in self.obstacles:
           grid[x][y] = "#"

       # goal
       gx, gy = self.goal
       grid[gx][gy] = "G"

       # agent
       x, y = self.state
       grid[x][y] = "A"

       for row in grid:
           print(" ".join(row))

       print("=" * 45)