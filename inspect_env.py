import gymnasium as gym
import stable_worldmodel
import time

env = gym.make("swm/TwoRoom-v1", render_mode="human")

try:
    observation, info = env.reset(seed=42)

    print("Simulator running. Press Ctrl+C in the terminal to stop.")

    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping simulator...")

finally:
    env.close()