
from racing_env import RacingEnv

env = RacingEnv(render_mode="human")

observation, info = env.reset()

print("Observation shape:", observation.shape)
print("Initial observation:", observation)

try:
    for step in range(600):
        # Temporary random actions, not a trained AI.
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        if step % 60 == 0:
            print(
                f"Step: {step}, "
                f"Reward: {reward:.2f}, "
                f"Speed: {info['speed']:.2f}, "
                f"Progress: {info['progress']:.2%}, "
                f"Collision: {info['collision']}"
            )

        if terminated or truncated:
            print("Episode ended:", info)
            observation, info = env.reset()

finally:
    env.close()
