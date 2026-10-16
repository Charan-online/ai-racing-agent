
from stable_baselines3 import PPO
from racing_env import RacingEnv


# 1. Load the best model saved during evaluation.
model = PPO.load("models/best/best_model.zip")

# 2. Create the racing environment with a visible window.
env = RacingEnv(render_mode="human")

# 3. Reset the environment.
observation, info = env.reset(seed=42)

total_reward = 0.0
steps = 0

print("Starting AI racing test...")
print("Close the racing window to stop viewing.")

try:
    while True:

        # Let the trained AI choose an action.
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        # Execute the action.
        observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        total_reward += reward
        steps += 1

        # Print progress every 100 steps.
        if steps % 100 == 0:
            print(
                f"Steps: {steps} | "
                f"Speed: {info['speed']:.2f} | "
                f"Progress: {info['progress'] * 100:.1f}% | "
                f"Reward: {total_reward:.2f}"
            )

        if terminated or truncated:
            break

    print("\n========== RACE RESULTS ==========")
    print("Steps:", steps)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Lap progress: {info['progress'] * 100:.1f}%")
    print("Collision:", info["collision"])

    if info["progress"] >= 1.0 and not info["collision"]:
        print("RESULT: SUCCESS! The AI completed a lap!")
    else:
        print("RESULT: The AI has not completed a clean lap yet.")

finally:
    env.close()
