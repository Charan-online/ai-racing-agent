
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from racing_env import RacingEnv


# -----------------------------------
# Create separate folders for the
# randomized-start training experiment
# -----------------------------------

os.makedirs("models/randomized_best", exist_ok=True)
os.makedirs("logs/randomized_eval", exist_ok=True)


# -----------------------------------
# Create environments
# reset() now randomizes the start
# during both training and evaluation
# -----------------------------------

train_env = Monitor(
    RacingEnv(render_mode=None)
)

eval_env = Monitor(
    RacingEnv(render_mode=None)
)


# -----------------------------------
# Evaluate the model periodically
# -----------------------------------

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/randomized_best/",
    log_path="logs/randomized_eval/",
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)


# -----------------------------------
# Configure PPO
# -----------------------------------

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    seed=42
)


# -----------------------------------
# Train the new agent
# -----------------------------------

TOTAL_TIMESTEPS = 100_000

print("\nTraining PPO with randomized starting conditions...")
print("Total timesteps:", TOTAL_TIMESTEPS)

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=False
    )

    model.save("models/racing_ppo_randomized_final")

    print("\nRandomized-start training completed!")
    print(
        "Final model: "
        "models/racing_ppo_randomized_final.zip"
    )
    print(
        "Best evaluated model: "
        "models/randomized_best/best_model.zip"
    )

finally:
    train_env.close()
    eval_env.close()
