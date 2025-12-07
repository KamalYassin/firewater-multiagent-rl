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

TILE_SIZE = 48
FPS = 1

BLACK = (10, 10, 15)
WHITE = (240, 240, 240)
GRID = (70, 70, 80)
FLOOR = (40, 40, 45)
WALL = (90, 90, 100)
LAVA = (220, 80, 60)
WATER = (70, 120, 230)
BLOCK = (150, 105, 70)
SWITCH = (240, 210, 80)
FIRE_AGENT = (230, 70, 70)
WATER_AGENT = (70, 140, 240)
FIRE_EXIT = (255, 140, 140)
WATER_EXIT = (140, 190, 255)
DOOR_CLOSED = (40, 160, 90)
DOOR_OPEN = (100, 220, 140)

LEGEND_LINES = [
    "Fire = red F, Fire exit = light red framed tile",
    "Water = blue G, Water exit = light blue framed tile",
    "Yellow = switch, Brown = block, Green = door",
]

CHAR_COLORS = {
    "#": WALL,
    ".": FLOOR,
    "W": WATER,
    "L": LAVA,
    "X": BLOCK,
    "F": FIRE_AGENT,
    "G": WATER_AGENT,
    "f": FIRE_EXIT,
    "g": WATER_EXIT,
    "1": SWITCH,
    "2": SWITCH,
    "3": SWITCH,
    "4": SWITCH,
    "5": SWITCH,
    "A": DOOR_CLOSED,
    "B": DOOR_CLOSED,
    "C": DOOR_CLOSED,
    "D": DOOR_CLOSED,
    "E": DOOR_CLOSED,
    "a": DOOR_OPEN,
    "b": DOOR_OPEN,
    "c": DOOR_OPEN,
    "d": DOOR_OPEN,
    "e": DOOR_OPEN,
}


def get_ascii_lines(env: FireWaterEnv):
    buf = io.StringIO()
    render_ascii(env, file=buf)
    text = buf.getvalue()
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return lines


def draw_env(screen, env: FireWaterEnv, tile_font, legend_font):
    lines = get_ascii_lines(env)
    rows = len(lines)
    cols = len(lines[0]) if rows > 0 else 0

    screen.fill(BLACK)

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            base_color = CHAR_COLORS.get(ch, FLOOR)
            rect = pygame.Rect(
                x * TILE_SIZE,
                y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )

            pygame.draw.rect(screen, base_color, rect)
            pygame.draw.rect(screen, GRID, rect, 1)

            glyph = None
            glyph_color = WHITE

            if ch == "F":
                glyph = "F"
            elif ch == "G":
                glyph = "G"
            elif ch in "12345":
                glyph = ch
                glyph_color = BLACK
            elif ch == "X":
                glyph = "X"
            elif ch == "f":
                pygame.draw.rect(screen, FIRE_EXIT, rect)
                pygame.draw.rect(screen, WHITE, rect, 3)
                glyph = "F"
            elif ch == "g":
                pygame.draw.rect(screen, WATER_EXIT, rect)
                pygame.draw.rect(screen, WHITE, rect, 3)
                glyph = "G"
            elif ch in "ABCDEabcde":
                door_color = DOOR_OPEN if ch.islower() else DOOR_CLOSED
                pygame.draw.rect(screen, door_color, rect)
                pygame.draw.rect(screen, WHITE, rect, 2)
                glyph = "D"

            if glyph is not None:
                text_surf = tile_font.render(glyph, True, glyph_color)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

    base_y = rows * TILE_SIZE + 4
    for i, line in enumerate(LEGEND_LINES):
        text_surf = legend_font.render(line, True, WHITE)
        screen.blit(text_surf, (8, base_y + i * (legend_font.get_linesize())))


def key_to_actions(pyg_key):
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

    tile_font = pygame.font.SysFont("Arial", 22, bold=True)
    legend_font = pygame.font.SysFont("Arial", 14)

    legend_widths = []
    legend_heights = []
    for line in LEGEND_LINES:
        surf = legend_font.render(line, True, WHITE)
        legend_widths.append(surf.get_width())
        legend_heights.append(surf.get_height())
    max_legend_width = max(legend_widths) if legend_widths else 0
    total_legend_height = sum(legend_heights) if legend_heights else 0

    width = max(cols * TILE_SIZE, max_legend_width + 16)
    height = rows * TILE_SIZE + total_legend_height + 8
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

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

        draw_env(screen, env, tile_font, legend_font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Pygame viewer for FireWaterEnv")
    parser.add_argument("level", help="Path to level .txt file")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--mode", choices=["manual", "policy"], default="manual")
    parser.add_argument("--policy-module", help="e.g. 'RL.easy_demo_policy' when using --mode policy")
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
