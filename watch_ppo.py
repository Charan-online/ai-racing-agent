
import time
import pygame
from stable_baselines3 import PPO
from racing_env import RacingEnv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/randomized_best/best_model.zip"

SIMULATION_FPS = 60
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

BACKGROUND = (15, 19, 29)
PANEL = (23, 29, 43)
WHITE = (242, 246, 255)
MUTED = (153, 165, 187)
GREEN = (65, 220, 145)
BLUE = (75, 155, 255)
ORANGE = (255, 183, 77)
RED = (255, 90, 100)


# --------------------------------------------------
# DASHBOARD DRAWING
# --------------------------------------------------

def draw_text(surface, text, font, color, x, y):
    image = font.render(str(text), True, color)
    surface.blit(image, (x, y))


def draw_hud(
    surface,
    fonts,
    race_number,
    steps,
    speed,
    max_speed,
    progress,
    best_lap,
    status,
    status_color,
):
    # Header
    pygame.draw.rect(
        surface, BACKGROUND,
        (0, 0, WINDOW_WIDTH, 88)
    )

    draw_text(
        surface, "AI RACING",
        fonts["title"], WHITE, 22, 12
    )

    draw_text(
        surface, "PPO AUTONOMOUS DRIVER",
        fonts["small"], MUTED, 24, 49
    )

    draw_text(
        surface, f"RACE {race_number:02d}",
        fonts["medium"], WHITE, 330, 29
    )

    pygame.draw.circle(
        surface, status_color, (650, 39), 6
    )

    draw_text(
        surface, status,
        fonts["medium"], status_color, 665, 27
    )

    # Footer
    pygame.draw.rect(
        surface, BACKGROUND,
        (0, 510, WINDOW_WIDTH, 90)
    )

    lap_time = steps / SIMULATION_FPS
    progress = max(0.0, min(1.0, float(progress)))
    speed_ratio = max(
        0.0,
        min(1.0, speed / max_speed if max_speed > 0 else 0.0)
    )

    # Lap time
    draw_text(
        surface, "LAP TIME",
        fonts["small"], MUTED, 22, 522
    )

    draw_text(
        surface, f"{lap_time:.2f} s",
        fonts["large"], WHITE, 22, 543
    )

    # Speed
    draw_text(
        surface, "SPEED",
        fonts["small"], MUTED, 240, 522
    )

    draw_text(
        surface, f"{speed:.1f}",
        fonts["large"], BLUE, 240, 543
    )

    draw_text(
        surface, f"/ {max_speed:.1f}",
        fonts["small"], MUTED, 298, 553
    )

    pygame.draw.rect(
        surface, PANEL,
        (240, 578, 145, 5), border_radius=3
    )

    pygame.draw.rect(
        surface, BLUE,
        (240, 578, int(145 * speed_ratio), 5),
        border_radius=3
    )

    # Progress
    draw_text(
        surface, "LAP PROGRESS",
        fonts["small"], MUTED, 435, 522
    )

    draw_text(
        surface, f"{progress * 100:.1f}%",
        fonts["large"], GREEN, 435, 543
    )

    pygame.draw.rect(
        surface, PANEL,
        (435, 578, 175, 5), border_radius=3
    )

    pygame.draw.rect(
        surface, GREEN,
        (435, 578, int(175 * progress), 5),
        border_radius=3
    )

    # Best lap
    draw_text(
        surface, "BEST LAP",
        fonts["small"], MUTED, 680, 522
    )

    best_text = (
        f"{best_lap:.2f} s"
        if best_lap is not None else "--"
    )

    draw_text(
        surface, best_text,
        fonts["large"], ORANGE, 680, 543
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    print("Loading trained PPO racing agent...")

    model = PPO.load(MODEL_PATH)
    env = RacingEnv(render_mode="human")

    # Disable display updates inside env.render().
    # The dashboard will perform the only display refresh.
    env.defer_display_update = True

    pygame.font.init()

    fonts = {
        "title": pygame.font.SysFont(
            "segoeui", 27, bold=True
        ),
        "medium": pygame.font.SysFont(
            "segoeui", 18, bold=True
        ),
        "small": pygame.font.SysFont(
            "segoeui", 13
        ),
        "large": pygame.font.SysFont(
            "segoeui", 24, bold=True
        ),
    }

    clock = pygame.time.Clock()
    race_number = 1
    best_lap = None
    running = True

    try:
        observation, info = env.reset(seed=42)

        steps = 0
        total_reward = 0.0

        print("Dashboard started!")
        print("Close the Pygame window or press Ctrl+C to stop.")

        while running:
            # Handle window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            # Preserve the trained AI's decision-making
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(int(action))

            steps += 1
            total_reward += float(reward)

            # Draw track and car without presenting the frame
            env.render()

            surface = pygame.display.get_surface()

            if surface is not None:
                speed = abs(
                    float(getattr(env, "speed", 0.0))
                )

                max_speed = float(
                    getattr(env, "MAX_SPEED", 5.0)
                )

                progress = float(
                    info.get("progress", 0.0)
                )

                status = "RACING"
                status_color = GREEN

                if terminated or truncated:
                    collision = info.get("collision", False)

                    if collision:
                        status = "CRASH"
                        status_color = RED
                    elif progress >= 1.0:
                        status = "LAP COMPLETE"
                        status_color = GREEN
                    else:
                        status = "STOPPED"
                        status_color = ORANGE

                # Draw dashboard on the same frame
                draw_hud(
                    surface=surface,
                    fonts=fonts,
                    race_number=race_number,
                    steps=steps,
                    speed=speed,
                    max_speed=max_speed,
                    progress=progress,
                    best_lap=best_lap,
                    status=status,
                    status_color=status_color,
                )

                # One display refresh after all drawing is complete
                pygame.display.flip()

            # Race finished?
            if terminated or truncated:
                lap_time = steps / SIMULATION_FPS
                progress = float(
                    info.get("progress", 0.0)
                )
                collision = info.get("collision", False)

                clean_lap = (
                    not collision and progress >= 1.0
                )

                if clean_lap:
                    if best_lap is None or lap_time < best_lap:
                        best_lap = lap_time

                    print(f"\nRace {race_number}: CLEAN LAP")
                else:
                    print(f"\nRace {race_number}: NOT COMPLETED")

                print(f"Steps: {steps}")
                print(f"Progress: {progress * 100:.1f}%")
                print(f"Lap time: {lap_time:.2f} seconds")
                print(f"Total reward: {total_reward:.2f}")

                # Keep the result visible briefly
                time.sleep(1)

                race_number += 1

                observation, info = env.reset(
                    seed=42 + race_number
                )

                steps = 0
                total_reward = 0.0

            clock.tick(SIMULATION_FPS)

    except KeyboardInterrupt:
        print("\nRacing stopped by user.")

    finally:
        env.close()
        pygame.quit()
        print("Racing dashboard closed.")


if __name__ == "__main__":
    main()
