import sys
import os
import io
import argparse
from importlib import import_module

import pygame

from .firewater_env import (
    FireWaterEnv,
    parse_level_from_file,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    render_ascii,
)


# ---------------------------------------------------------
# Basic Pygame config
# ---------------------------------------------------------

TILE_SIZE = 40      
FPS = 1      

BLACK   = (0, 0, 0)
WHITE   = (240, 240, 240)
GREY    = (120, 120, 120)
CYAN    = (0, 200, 200)      
RED     = (220, 60, 60)
BLUE    = (60, 120, 220)
GREEN   = (60, 200, 60)
YELLOW  = (240, 220, 70)
BROWN   = (140, 100, 60)
PURPLE  = (150, 70, 200)

CHAR_COLORS = {
    '#': GREY,      
    '.': BLACK,     
    'W': BLUE,      
    'L': RED,       
    'X': BROWN,    
    'F': RED,       
    'G': BLUE,      
    'f': PURPLE,    
    'g': CYAN,      
    '1': YELLOW,
    '2': YELLOW,
    '3': YELLOW,
    '4': YELLOW,
    '5': YELLOW,
}


def get_ascii_lines(env: FireWaterEnv):
    buf = io.StringIO()
    render_ascii(env, file=buf)
    text = buf.getvalue()
    # drop trailing blank line render_ascii prints
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return lines



def draw_env(screen, env: FireWaterEnv, font):
    lines = get_ascii_lines(env)
    rows = len(lines)
    cols = len(lines[0]) if rows > 0 else 0

    screen.fill(BLACK)

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            color = CHAR_COLORS.get(ch, BLACK)
            rect = pygame.Rect(
                x * TILE_SIZE,
                y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )
            pygame.draw.rect(screen, color, rect)

            pygame.draw.rect(screen, WHITE, rect, 1)

    info_text = "WASD = Fire, IJKL = Water, Q = quit"
    text_surf = font.render(info_text, True, WHITE)
    screen.blit(text_surf, (5, rows * TILE_SIZE + 5))


def key_to_actions(pyg_key):
    # manual controls
    fire_action = ACTION_STAY
    water_action = ACTION_STAY

    if pyg_key == pygame.K_w:
        fire_action = ACTION_UP
    elif pyg_key == pygame.K_a:
        fire_action = ACTION_LEFT
    elif pyg_key == pygame.K_s:
        fire_action = ACTION_DOWN
    elif pyg_key == pygame.K_d:
        fire_action = ACTION_RIGHT

    if pyg_key == pygame.K_i:
        water_action = ACTION_UP
    elif pyg_key == pygame.K_j:
        water_action = ACTION_LEFT
    elif pyg_key == pygame.K_k:
        water_action = ACTION_DOWN
    elif pyg_key == pygame.K_l:
        water_action = ACTION_RIGHT

    return fire_action, water_action


def run_viewer(level_path: str, max_steps: int = 200, policy_fn=None):
    lvl = parse_level_from_file(level_path)
    env = FireWaterEnv(lvl, max_steps=max_steps)
    obs = env.reset()

    lines = get_ascii_lines(env)
    rows = len(lines)
    cols = len(lines[0]) if rows > 0 else 0

    pygame.init()
    pygame.display.set_caption(f"Fire & Water Viewer - {os.path.basename(level_path)}")

    width = cols * TILE_SIZE
    height = rows * TILE_SIZE + 30
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    running = True
    done = False
    step = 0

    while running and not done and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                    break
                if policy_fn is None:
                    fire_a, water_a = key_to_actions(event.key)
                    obs, reward, done, info = env.step(fire_a, water_a)
                    step += 1

        if policy_fn is not None and running and not done:
            a_fire, a_water = policy_fn(obs)
            obs, reward, done, info = env.step(a_fire, a_water)
            step += 1

        draw_env(screen, env, font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Pygame viewer for FireWaterEnv")
    parser.add_argument("level", help="Path to level .txt file")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--mode", choices=["manual", "policy"], default="manual")
    parser.add_argument("--policy-module",
                        help="e.g. 'RL.easy_demo_policy' when using --mode policy")
    args = parser.parse_args()

    policy_fn = None
    if args.mode == "policy":
        if not args.policy_module:
            parser.error("--mode policy requires --policy-module MODULE_PATH")
        mod = import_module(args.policy_module)
        if not hasattr(mod, "policy_fn"):
            parser.error(f"Module {args.policy_module!r} has no function 'policy_fn'")
        policy_fn = getattr(mod, "policy_fn")

    run_viewer(args.level, max_steps=args.max_steps, policy_fn=policy_fn)


if __name__ == "__main__":
    main()
