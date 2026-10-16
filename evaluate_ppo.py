
import time
import statistics

from stable_baselines3 import PPO
from racing_env import RacingEnv


# -------------------------------
# Configuration
# -------------------------------

MODEL_PATH = "models/randomized_best/best_model.zip"
NUM_RACES = 20
SHOW_WINDOW = False
SIMULATION_FPS = 60


# -------------------------------
# Load the trained agent
# -------------------------------

print("Loading trained PPO model...")

model = PPO.load(MODEL_PATH)

render_mode = "human" if SHOW_WINDOW else None
env = RacingEnv(render_mode=render_mode)

results = []


# -------------------------------
# Run randomized races
# -------------------------------

try:
    for race_number in range(1, NUM_RACES + 1):

        # Each seed generates different starting conditions.
        observation, reset_info = env.reset(
            seed=1000 + race_number
        )

        total_reward = 0.0
        steps = 0
        start_time = time.perf_counter()

        while True:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        runtime = time.perf_counter() - start_time

        progress = float(info["progress"])
        collision = bool(info["collision"])

        clean_lap = progress >= 1.0 and not collision

        simulated_time = steps / SIMULATION_FPS

        result = {
            "race": race_number,
            "steps": steps,
            "progress": progress * 100,
            "collision": collision,
            "clean_lap": clean_lap,
            "reward": total_reward,
            "simulated_time": simulated_time,
            "runtime": runtime,
            "start_angle": reset_info["start_angle"],
            "lateral_offset": reset_info["lateral_offset"],
            "heading_offset": reset_info["heading_offset"],
            "start_speed": reset_info["start_speed"],
        }

        results.append(result)

        status = "CLEAN LAP" if clean_lap else "FAILED"

        print(
            f"Race {race_number:02d}/{NUM_RACES} | "
            f"{status} | "
            f"Progress: {progress * 100:.1f}% | "
            f"Steps: {steps} | "
            f"Start speed: {reset_info['start_speed']:.2f}"
        )

finally:
    env.close()


# -------------------------------
# Calculate performance metrics
# -------------------------------

successful = [
    result for result in results
    if result["clean_lap"]
]

failed = [
    result for result in results
    if not result["clean_lap"]
]

success_rate = len(successful) / NUM_RACES * 100

print("\n" + "=" * 55)
print("       RANDOMIZED PPO EVALUATION REPORT")
print("=" * 55)

print(f"Total races: {NUM_RACES}")
print(f"Clean laps: {len(successful)}")
print(f"Failed laps: {len(failed)}")
print(f"Clean-lap success rate: {success_rate:.1f}%")

if successful:

    lap_times = [
        result["simulated_time"]
        for result in successful
    ]

    lap_steps = [
        result["steps"]
        for result in successful
    ]

    print(
        f"Average steps per clean lap: "
        f"{statistics.mean(lap_steps):.1f}"
    )

    print(
        f"Average simulated lap time: "
        f"{statistics.mean(lap_times):.2f} seconds"
    )

    print(
        f"Best simulated lap time: "
        f"{min(lap_times):.2f} seconds"
    )

    print(
        f"Worst successful lap time: "
        f"{max(lap_times):.2f} seconds"
    )

else:
    print("No successful laps; lap-time statistics unavailable.")


print("\nAverage computer runtime per race: "
      f"{statistics.mean(r['runtime'] for r in results):.3f} seconds")


# -------------------------------
# Print randomized start conditions
# -------------------------------

print("\nStarting conditions:")
print("-" * 55)

for result in results:
    print(
        f"Race {result['race']:02d} | "
        f"Position angle: {result['start_angle']:+.3f} rad | "
        f"Offset: {result['lateral_offset']:+.1f} px | "
        f"Heading offset: {result['heading_offset']:+.3f} rad | "
        f"Start speed: {result['start_speed']:.2f}"
    )


print("\nEvaluation completed!")
