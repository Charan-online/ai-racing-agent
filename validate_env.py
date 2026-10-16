
from gymnasium.utils.env_checker import check_env
from racing_env import RacingEnv

print("Checking Gymnasium environment...")

env = RacingEnv(render_mode=None)

check_env(env, warn=True)

print("Environment API check passed!")

observation, info = env.reset(seed=42)

print("Observation shape:", observation.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

assert env.observation_space.contains(observation)

print("Testing all 9 actions...")

for action in range(9):
    observation, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(observation)

print("All actions tested successfully!")

print("Testing episode termination...")

observation, info = env.reset(seed=123)

for step in range(5000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended after", step + 1, "steps")
        print("Final reward:", reward)
        print("Collision:", info["collision"])
        break
else:
    print("Episode did not finish within 5000 steps.")

env.close()

print("Validation complete!")
