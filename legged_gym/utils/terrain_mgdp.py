import random

import numpy as np
from isaacgym import terrain_utils


MGDP_PARKOUR_TERRAIN_NAMES = (
    "single_gap",
    "step_stone",
    "two_row_stones",
    "one_row_stones",
    "single_bridge",
    "air_beams",
    "air_stones",
    "hurdle",
    "ramp",
    "corridor",
    "stairs_up",
    "flat",
    "rough_flat",
)


def _normalized_difficulty(difficulty, cfg):
    if cfg is None:
        max_difficulty = 0.9
    else:
        max_difficulty = float(
            getattr(
                cfg,
                "mgdp_max_difficulty",
                (cfg.num_rows - 1) / cfg.num_rows if getattr(cfg, "num_rows", 0) > 1 else 0.9,
            )
        )
    max_difficulty = max(max_difficulty, 1.0e-6)
    return float(np.clip(difficulty / max_difficulty, 0.0, 1.0))


def _interpolate_difficulty(difficulty, cfg, low, high):
    normalized_difficulty = _normalized_difficulty(difficulty, cfg)
    return low + (high - low) * normalized_difficulty


def build_mgdp_parkour_terrain(terrain, choice, difficulty, cfg, proportions):
    depth = float(getattr(cfg, "mgdp_depth", 0.6))
    normalized_difficulty = _normalized_difficulty(difficulty, cfg)
    step_height = 0.05 + 0.18 * difficulty

    if choice < proportions[0]:
        terrain.terrain_name = "single_gap"
        terrain.terrain_id = 0
        parkour_step_gap_terrain(terrain, difficulty=difficulty, depth=depth, cfg=cfg, platform_size=2.0)
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[1]:
        terrain.terrain_name = "step_stone"
        terrain.terrain_id = 1
        stone_size = 0.5 - 0.3 * normalized_difficulty
        stone_distance_x = 0.175 + 0.075 * normalized_difficulty
        stone_distance_y = 0.1 + 0.075 * normalized_difficulty
        stone_jitter_scale = 0.48 * normalized_difficulty
        stepping_stones_terrain(
            terrain,
            stone_size=stone_size,
            stone_distance_x=stone_distance_x,
            stone_distance_y=stone_distance_y,
            stone_jitter_x=stone_jitter_scale * stone_distance_x,
            stone_jitter_y=stone_jitter_scale * stone_distance_y,
            row_x_jitter=stone_jitter_scale * stone_distance_x,
            max_height=step_height,
            platform_size=2.0,
            depth=depth,
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[2]:
        terrain.terrain_name = "two_row_stones"
        terrain.terrain_id = 2
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        stepping_two_stones_terrain(
            terrain,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth,
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[3]:
        terrain.terrain_name = "one_row_stones"
        terrain.terrain_id = 3
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        gap_range = getattr(cfg, "one_row_stone_gap_range", [0.05, 0.35])
        gap_low = float(gap_range[0])
        gap_high = float(gap_range[1])
        if gap_high < gap_low:
            gap_low, gap_high = gap_high, gap_low
        min_stone_size_x = float(getattr(cfg, "one_row_stone_x_size", terrain.horizontal_scale))
        stone_distance = gap_low + (gap_high - gap_low) * normalized_difficulty
        stone_pitch_x = max(min_stone_size_x + gap_high, terrain.horizontal_scale)
        stone_size_x = max(terrain.horizontal_scale, stone_pitch_x - stone_distance)
        stepping_one_stones_terrain(
            terrain,
            difficulty=difficulty,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth,
            stone_size_x=stone_size_x,
            stone_pitch_x=stone_pitch_x,
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[4]:
        terrain.terrain_name = "single_bridge"
        terrain.terrain_id = 4
        bridge_size = 0.6 if difficulty < 0.2 else -0.6 * difficulty + 0.8
        stepping_one_bridge_terrain(
            terrain,
            stone_size=bridge_size,
            difficulty=difficulty,
            platform_size=2.0,
            depth=depth,
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[5]:
        terrain.terrain_name = "air_beams"
        terrain.terrain_id = 5
        stepping_air_beams_base_terrain(terrain, platform_size=2.0, depth=depth)
        terrain.extra_trimesh_specs = getattr(terrain, "extra_trimesh_specs", [])
        terrain.extra_trimesh_specs.append(
            {
                "type": "air_beams",
                "difficulty": float(difficulty),
                "platform_size": 2.0,
                "depth": depth,
            }
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[6]:
        terrain.terrain_name = "air_stones"
        terrain.terrain_id = 6
        air_stone_base_terrain(terrain, depth=depth)
        terrain.extra_trimesh_specs = getattr(terrain, "extra_trimesh_specs", [])
        terrain.extra_trimesh_specs.append(
            {
                "type": "air_stones",
                "difficulty": float(difficulty),
                "platform_size": 2.0,
                "depth": depth,
            }
        )
    elif choice < proportions[7]:
        terrain.terrain_name = "hurdle"
        terrain.terrain_id = 7
        hurdle_height_min = step_height if difficulty < 0.1 else 0.15 + 0.55 * difficulty
        hurdle_height_max = 0.1 + step_height if difficulty < 0.1 else 0.2 + 0.55 * difficulty
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        parkour_hurdle_terrain(
            terrain,
            difficulty=difficulty,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            hurdle_height_range=[hurdle_height_min, hurdle_height_max],
            platform_size=2.0,
            depth=depth,
        )
        add_roughness(terrain, cfg, difficulty)
    elif choice < proportions[8]:
        terrain.terrain_name = "ramp"
        terrain.terrain_id = 8
        half_sloped_terrain(
            terrain,
            difficulty=difficulty,
            cfg=cfg,
            depth=depth,
            max_angle_deg=float(getattr(cfg, "ramp_max_angle_deg", 30.0)),
        )
    elif choice < proportions[9]:
        terrain.terrain_name = "corridor"
        terrain.terrain_id = 9
        corridor_half_width = _interpolate_difficulty(difficulty, cfg, 0.4, 0.20)
        narrow_corridor(
            terrain,
            corridor_half_width=corridor_half_width,
            depth=depth,
        )
    elif choice < proportions[10]:
        terrain.terrain_name = "stairs_up"
        terrain.terrain_id = 10
        stair_step_height = 0.05 + 0.20 * normalized_difficulty
        half_stairs_terrain(
            terrain,
            step_width=0.31,
            step_height=stair_step_height,
            platform_size=2.0,
            landing_length=2.0,
            final_platform_length=1.0,
            depth=depth,
        )
    elif len(proportions) <= 12 or choice < proportions[11]:
        terrain.terrain_name = "flat"
        terrain.terrain_id = 11
        flat_terrain(terrain)
    else:
        terrain.terrain_name = "rough_flat"
        terrain.terrain_id = 12
        rough_flat_terrain(terrain, cfg=cfg, difficulty=difficulty)

    return terrain


def add_roughness(terrain, cfg, difficulty):
    if not getattr(cfg, "mgdp_add_roughness", True):
        return
    height_range = getattr(cfg, "mgdp_roughness_height", [0.01, 0.04])
    low = float(height_range[0])
    high = float(height_range[1])
    if high <= 0.0 or high < low:
        return
    roughness = random.uniform(low, low + (high - low) * difficulty)
    if roughness <= 0.0:
        return
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-roughness,
        max_height=roughness,
        step=0.005,
        downsampled_scale=float(getattr(cfg, "mgdp_roughness_downsampled_scale", 0.5)),
    )


def flat_terrain(terrain):
    terrain.height_field_raw[:, :] = 0


def rough_flat_terrain(terrain, cfg=None, difficulty=0.0):
    flat_terrain(terrain)
    normalized_difficulty = _normalized_difficulty(difficulty, cfg)

    height_range = getattr(cfg, "mgdp_rough_flat_spike_height", [0.015, 0.055])
    height_low = max(0.0, float(height_range[0]))
    height_high = max(height_low, float(height_range[1]))
    max_height = height_low + (height_high - height_low) * normalized_difficulty

    density_range = getattr(cfg, "mgdp_rough_flat_spike_density", [5.0, 18.0])
    density_low = max(0.0, float(density_range[0]))
    density_high = max(density_low, float(density_range[1]))
    spike_density = density_low + (density_high - density_low) * normalized_difficulty

    radius_range = getattr(cfg, "mgdp_rough_flat_spike_radius", [0.025, 0.075])
    radius_low = max(terrain.horizontal_scale, float(radius_range[0]))
    radius_high = max(radius_low, float(radius_range[1]))

    platform_length = float(getattr(cfg, "start_platform_length", 2.0))
    platform_width = float(getattr(cfg, "start_platform_width", 2.0))
    platform_x2 = max(1, min(terrain.width, int(platform_length / terrain.horizontal_scale)))
    center_y = terrain.length // 2
    platform_half_width = max(1, int(0.5 * platform_width / terrain.horizontal_scale))
    platform_y1 = max(0, center_y - platform_half_width)
    platform_y2 = min(terrain.length, center_y + platform_half_width)

    rough_x1 = min(terrain.width - 1, platform_x2)
    rough_x2 = terrain.width
    margin_y = max(1, int(0.10 / terrain.horizontal_scale))
    rough_y1 = min(terrain.length - 1, margin_y)
    rough_y2 = max(rough_y1 + 1, terrain.length - margin_y)
    rough_area = (
        max(terrain.horizontal_scale, (rough_x2 - rough_x1) * terrain.horizontal_scale)
        * max(terrain.horizontal_scale, (rough_y2 - rough_y1) * terrain.horizontal_scale)
    )
    spike_count = int(round(spike_density * rough_area))
    if spike_count <= 0 or max_height <= 0.0:
        return

    for _ in range(spike_count):
        center_x = np.random.randint(rough_x1, rough_x2)
        center_y = np.random.randint(rough_y1, rough_y2)
        radius_m = np.random.uniform(radius_low, radius_high)
        radius_px = max(1, int(round(radius_m / terrain.horizontal_scale)))
        peak_height = np.random.uniform(height_low, max_height)
        peak_raw = max(1, int(round(peak_height / terrain.vertical_scale)))

        x1 = max(0, center_x - radius_px)
        x2 = min(terrain.width, center_x + radius_px + 1)
        y1 = max(0, center_y - radius_px)
        y2 = min(terrain.length, center_y + radius_px + 1)
        for x in range(x1, x2):
            for y in range(y1, y2):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist > radius_px:
                    continue
                height_raw = int(round(peak_raw * (1.0 - dist / (radius_px + 1.0))))
                if height_raw > terrain.height_field_raw[x, y]:
                    terrain.height_field_raw[x, y] = height_raw

    terrain.height_field_raw[:platform_x2, platform_y1:platform_y2] = 0


def narrow_corridor(terrain, corridor_half_width, depth=2.0, wall_height=None):
    platform_size = int(2 / terrain.horizontal_scale)
    base_height = int(-depth / terrain.vertical_scale)
    if wall_height is None:
        wall_height = np.random.randint(200, 251)
    else:
        wall_height = max(0, int(round(wall_height / terrain.vertical_scale)))

    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0

    center_y = terrain.length // 2
    corridor_half_width = max(1, int(round(corridor_half_width / terrain.horizontal_scale)))
    corridor_half_width = int(np.clip(corridor_half_width, 1, max(1, (end_y - start_y) // 2 - 1)))

    terrain.height_field_raw[platform_size + 20 :, center_y + corridor_half_width : end_y] = wall_height
    terrain.height_field_raw[platform_size + 20 :, start_y : center_y - corridor_half_width] = wall_height


def half_sloped_terrain(
    terrain,
    difficulty,
    cfg=None,
    platform_size=2.0,
    slope_strength=None,
    final_platform_length=1.0,
    depth=2.0,
    max_angle_deg=30.0,
):
    platform_size = int(platform_size / terrain.horizontal_scale)
    if slope_strength is None:
        angle_deg = _normalized_difficulty(difficulty, cfg) * max_angle_deg
        slope_strength = np.tan(np.deg2rad(angle_deg)) * terrain.horizontal_scale / terrain.vertical_scale
    final_platform_length = max(1, int(final_platform_length / terrain.horizontal_scale))

    base_height = int(-depth / terrain.vertical_scale)
    start_y, end_y = 20, terrain.length - 20

    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0

    up_start = platform_size
    up_end = min(2 * platform_size, terrain.width)
    xs = np.arange(up_start, up_end)
    max_height = slope_strength * (up_end - up_start)
    up_heights = (slope_strength * (xs - up_start)).astype(np.int16)
    terrain.height_field_raw[up_start:up_end, start_y:end_y] = up_heights[:, None]

    mid_start = up_end
    mid_end = min(mid_start + platform_size, terrain.width)
    terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = max_height

    down_start = mid_end
    down_end = min(down_start + (up_end - up_start), terrain.width - final_platform_length)
    xs = np.arange(down_start, down_end)
    down_heights = (max_height - slope_strength * (xs - down_start)).astype(np.int16)
    terrain.height_field_raw[down_start:down_end, start_y:end_y] = down_heights[:, None]
    terrain.height_field_raw[down_end:terrain.width, start_y:end_y] = 0


def half_stairs_terrain(
    terrain,
    step_width,
    step_height,
    platform_size=2.0,
    landing_length=2.0,
    final_platform_length=1.0,
    depth=2.0,
):
    step_width = max(1, int(step_width / terrain.horizontal_scale))
    step_height = max(1, int(step_height / terrain.vertical_scale))
    platform_size = max(1, int(platform_size / terrain.horizontal_scale))
    landing_length = max(1, int(landing_length / terrain.horizontal_scale))
    final_platform_length = max(1, int(final_platform_length / terrain.horizontal_scale))

    base_height = int(-depth / terrain.vertical_scale)
    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0

    up_start = platform_size
    up_end = min(2 * platform_size, terrain.width)
    num_up_steps = max(1, (up_end - up_start) // step_width)
    current_height = 0
    x_cursor = up_start
    for _ in range(num_up_steps):
        next_cursor = min(x_cursor + step_width, up_end)
        current_height += step_height
        terrain.height_field_raw[x_cursor:next_cursor, start_y:end_y] = current_height
        x_cursor = next_cursor
    top_height = current_height
    if x_cursor < up_end:
        terrain.height_field_raw[x_cursor:up_end, start_y:end_y] = top_height

    mid_start = up_end
    mid_end = min(mid_start + landing_length, terrain.width)
    terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = top_height

    down_start = mid_end
    down_end = min(down_start + (up_end - up_start), terrain.width - final_platform_length)
    num_down_steps = max(1, (down_end - down_start) // step_width)
    x_cursor = down_start
    for step_idx in range(num_down_steps):
        next_cursor = min(x_cursor + step_width, down_end)
        step_level = max(0, top_height - step_height * (step_idx + 1))
        terrain.height_field_raw[x_cursor:next_cursor, start_y:end_y] = step_level
        x_cursor = next_cursor
    terrain.height_field_raw[down_end:terrain.width, start_y:end_y] = 0


def parkour_step_gap_terrain(terrain, difficulty, depth, cfg=None, platform_size=2.0):
    normalized_difficulty = _normalized_difficulty(difficulty, cfg)
    gap_width_range = getattr(cfg, "mgdp_gap_width_range", [0.1, 0.9]) if cfg is not None else [0.1, 0.9]
    gap_width_min = float(gap_width_range[0])
    gap_width_max = float(gap_width_range[1])
    if gap_width_max < gap_width_min:
        gap_width_min, gap_width_max = gap_width_max, gap_width_min

    gap_width = gap_width_min + (gap_width_max - gap_width_min) * normalized_difficulty
    gap_size = max(1, int(round(gap_width / terrain.horizontal_scale)))
    depth = int(depth / terrain.vertical_scale)

    platform_size = int(platform_size / terrain.horizontal_scale)
    start_y = 0
    end_y = int(terrain.length - platform_size / 8)

    start_x = platform_size
    center_x = terrain.width // 2
    terrain.height_field_raw[start_x:center_x, start_y:end_y] = -depth
    terrain.height_field_raw[start_x + gap_size : center_x - gap_size, start_y + gap_size : end_y - gap_size] = 0

    start_x = center_x + int(platform_size / 2)
    end_x = int(terrain.width)
    terrain.height_field_raw[start_x:end_x, start_y:end_y] = -depth
    terrain.height_field_raw[start_x + gap_size : end_x - gap_size, start_y + gap_size : end_y - gap_size] = 0


def stepping_stones_terrain(
    terrain,
    stone_size,
    stone_distance_x,
    stone_distance_y,
    stone_jitter_x,
    stone_jitter_y,
    row_x_jitter,
    max_height,
    platform_size=1.0,
    depth=1.0,
):
    stone_size = max(1, int(round(stone_size / terrain.horizontal_scale)))
    stone_distance_x = max(1, int(round(stone_distance_x / terrain.horizontal_scale)))
    stone_distance_y = max(1, int(round(stone_distance_y / terrain.horizontal_scale)))
    stone_jitter_x = max(0, int(round(stone_jitter_x / terrain.horizontal_scale)))
    stone_jitter_y = max(0, int(round(stone_jitter_y / terrain.horizontal_scale)))
    row_x_jitter = max(0, int(round(row_x_jitter / terrain.horizontal_scale)))
    max_height = max(1, min(int(round(max_height / terrain.vertical_scale)), 40))
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(1, max_height + 1, step=4)
    pitch_y = stone_size + stone_distance_y
    num_rows = max(1, (terrain.length + stone_distance_y) // pitch_y)
    pattern_height = num_rows * stone_size + (num_rows - 1) * stone_distance_y
    base_row_y = max(0, (terrain.length - pattern_height) // 2)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    for row_idx in range(num_rows):
        row_y = base_row_y + row_idx * pitch_y
        row_shift_x = random.randint(-row_x_jitter, row_x_jitter) if row_x_jitter > 0 else 0
        start_x = platform_size + row_shift_x
        while start_x < terrain.width:
            stone_shift_x = random.randint(-stone_jitter_x, stone_jitter_x) if stone_jitter_x > 0 else 0
            stone_shift_y = random.randint(-stone_jitter_y, stone_jitter_y) if stone_jitter_y > 0 else 0
            stone_start_x = int(np.clip(start_x + stone_shift_x, platform_size, max(platform_size, terrain.width - stone_size)))
            stone_start_y = int(np.clip(row_y + stone_shift_y, 0, max(0, terrain.length - stone_size)))
            stop_x = min(terrain.width, stone_start_x + stone_size)
            stop_y = min(terrain.length, stone_start_y + stone_size)
            terrain.height_field_raw[stone_start_x:stop_x, stone_start_y:stop_y] = np.random.choice(height_range)
            start_x += stone_size + stone_distance_x

    x1 = 0
    x2 = platform_size
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def stepping_two_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=1.0):
    stone_size = int(stone_size / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    max_height = int(np.clip(max_height, 0, 30))
    platform_size = int(platform_size / terrain.horizontal_scale)

    height_range = np.arange(1, max_height, step=3)
    stone_size = int(np.clip(stone_size, 6, 16))

    stone_distance_gap = int(stone_distance / terrain.horizontal_scale)
    stone_distance_gap = int(np.clip(stone_distance_gap, 1, 6))
    stone_distance_range = np.arange(0, stone_distance_gap, step=1) + 1

    platform_y = (terrain.length - platform_size) // 2
    track_center_y = platform_y + platform_size // 2
    lateral_center_offset = int(round(0.22 / terrain.horizontal_scale))
    row1_y = int(np.clip(track_center_y - lateral_center_offset - stone_size / 2, 0, max(0, terrain.length - stone_size)))
    row2_y = int(np.clip(track_center_y + lateral_center_offset - stone_size / 2, 0, max(0, terrain.length - stone_size)))

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x:stop_x, row1_y : row1_y + stone_size] = np.random.choice(height_range)
        terrain.height_field_raw[start_x:stop_x, row2_y : row2_y + stone_size] = np.random.choice(height_range)
        stone_distance_gap = np.random.choice(stone_distance_range)
        start_x += stone_size + stone_distance_gap

    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size] = 0
    return terrain


def stepping_one_stones_terrain(
    terrain,
    difficulty,
    stone_size,
    stone_distance,
    max_height,
    platform_size=1.0,
    depth=1.0,
    stone_size_x=None,
    stone_pitch_x=None,
):
    if stone_size_x is None:
        stone_size_x = terrain.horizontal_scale
    stone_size_x = max(float(terrain.horizontal_scale), float(stone_size_x))
    if stone_pitch_x is None:
        stone_pitch_x = stone_size_x + float(stone_distance)
    stone_pitch_x = max(float(stone_pitch_x), stone_size_x + float(terrain.horizontal_scale))
    max_height = int(max_height / terrain.vertical_scale)
    max_height = int(np.clip(max_height, 0, 20))
    platform_size_cells = int(platform_size / terrain.horizontal_scale)

    if difficulty < 0.3:
        height_range = np.arange(1, max_height, step=3)
    elif difficulty < 0.7:
        height_range = np.arange(1, max_height, step=4)
    else:
        height_range = np.arange(1, max_height, step=12)

    platform_y = (terrain.length - platform_size_cells) // 2
    track_center_y = (platform_y + platform_size_cells / 2.0) * terrain.horizontal_scale
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    center_x = float(platform_size) + stone_pitch_x / 2.0
    terrain_width_m = terrain.width * terrain.horizontal_scale
    while center_x - stone_size_x / 2.0 < terrain_width_m:
        stone_size_y = float(np.random.choice(np.arange(12, 30, step=1))) * terrain.horizontal_scale
        stone_height = float(np.random.choice(height_range)) * terrain.vertical_scale
        _stamp_box_height(
            terrain.height_field_raw,
            center_position=np.array([center_x, track_center_y, 0.0], dtype=np.float32),
            size=np.array([stone_size_x, stone_size_y, 0.0], dtype=np.float32),
            horizontal_scale=terrain.horizontal_scale,
            vertical_scale=terrain.vertical_scale,
            stamp_height=stone_height,
        )
        center_x += stone_pitch_x

    x1 = 0
    x2 = platform_size_cells
    terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size_cells] = 0
    return terrain


def stepping_one_bridge_terrain(terrain, stone_size, difficulty, platform_size=1.0, depth=1.0):
    del difficulty

    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    row_y = int(platform_y + platform_size / 2 - stone_size / 2)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    height_range = np.arange(0, 10, step=2)

    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x:stop_x, row_y : row_y + stone_size] = np.random.choice(height_range)
        start_x += stone_size

    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size] = 0
    return terrain


def stepping_air_beams_base_terrain(terrain, platform_size=2.0, depth=1.0):
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    terrain.height_field_raw[:platform_size, platform_y : platform_y + platform_size] = 0
    return terrain


def air_stone_base_terrain(terrain, depth=1.0):
    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    terrain.height_field_raw[:, start_y:end_y] = 0
    return terrain


def stepping_beams_terrain(terrain, difficulty, beam_length, beam_gap, platform_size=2.0, depth=1.0):
    beam_length = int(beam_length / terrain.horizontal_scale)
    beam_gap = int(max(1, beam_gap / terrain.horizontal_scale))
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    start_x = 0
    min_beam_width = 8 if difficulty < 0.5 else 5
    max_beam_width = 18 if difficulty < 0.5 else 12
    while start_x < terrain.width:
        beam_width = random.randint(min_beam_width, max_beam_width)
        row_y = int(platform_y + platform_size / 2 - beam_width / 2)
        stop_x = min(terrain.width, start_x + max(2, beam_length))
        terrain.height_field_raw[start_x:stop_x, row_y : row_y + beam_width] = 0
        start_x += max(2, beam_length) + beam_gap

    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size] = 0
    return terrain


def air_stones_terrain(terrain, difficulty, platform_size=2.0, depth=1.0):
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    stone_len_min = 8 if difficulty < 0.4 else 6
    stone_len_max = 18 if difficulty < 0.4 else 12
    gap_min = 6 if difficulty < 0.4 else 8
    gap_max = 14 if difficulty < 0.4 else 18

    start_x = 0
    toggle = False
    while start_x < terrain.width:
        stone_len = random.randint(stone_len_min, stone_len_max)
        stone_width = random.randint(10, 22)
        y_offset = -8 if toggle else 8
        row_y = int(platform_y + platform_size / 2 - stone_width / 2 + y_offset)
        row_y = int(np.clip(row_y, 0, max(0, terrain.length - stone_width)))
        stop_x = min(terrain.width, start_x + stone_len)
        terrain.height_field_raw[start_x:stop_x, row_y : row_y + stone_width] = 0
        start_x += stone_len + random.randint(gap_min, gap_max)
        toggle = not toggle

    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y : platform_y + platform_size] = 0
    return terrain


def parkour_hurdle_terrain(
    terrain,
    difficulty,
    stone_size,
    stone_distance,
    max_height,
    hurdle_height_range,
    platform_size=1.0,
    depth=1.0,
):
    del stone_distance, max_height, depth

    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2

    terrain.height_field_raw[:, :] = int(-0.6 / terrain.vertical_scale)

    row0_y = int(platform_y + platform_size / 2 - 17 / 2)
    terrain.height_field_raw[:, row0_y : row0_y + 18] = 0

    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    step_height = np.random.randint(hurdle_height_min, hurdle_height_max)

    start_x = platform_size + 20
    while start_x < terrain.width - 20:
        stone_size_x = int(np.clip(stone_size, 12, 30))
        stone_size_y = np.random.choice(np.arange(17, 30, step=1))
        stop_x = min(terrain.width, start_x + stone_size_x)
        row_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
        terrain.height_field_raw[start_x:stop_x, row_y : row_y + stone_size_y] = step_height
        stone_distance_gap = int(np.clip(int(difficulty / terrain.horizontal_scale), 60, 80))
        start_x += stone_size + stone_distance_gap

    return terrain


def _box_trimesh(size, center_position, rpy=np.zeros(3)):
    if not np.all(rpy == 0):
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype=np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2
    vertices = np.round(vertices, 3)

    triangles = -np.ones((12, 3), dtype=np.uint32)
    triangles[0] = [0, 2, 1]
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2]
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6]
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5]
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4]
    triangles[9] = [1, 5, 4]
    triangles[10] = [2, 6, 3]
    triangles[11] = [3, 6, 7]
    return vertices, triangles


def _stamp_box_height(height_field_raw, center_position, size, horizontal_scale, vertical_scale, stamp_height=None):
    if stamp_height is None:
        stamp_height = center_position[2] + size[2] / 2
    height_value = int(np.round(stamp_height / vertical_scale))

    x0 = int(np.floor((center_position[0] - size[0] / 2) / horizontal_scale))
    x1 = int(np.ceil((center_position[0] + size[0] / 2) / horizontal_scale))
    y0 = int(np.floor((center_position[1] - size[1] / 2) / horizontal_scale))
    y1 = int(np.ceil((center_position[1] + size[1] / 2) / horizontal_scale))
    x0 = max(0, min(x0, height_field_raw.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, height_field_raw.shape[0]))
    y0 = max(0, min(y0, height_field_raw.shape[1] - 1))
    y1 = max(y0 + 1, min(y1, height_field_raw.shape[1]))
    height_field_raw[x0:x1, y0:y1] = height_value


def _build_air_beam_boxes(origin_x, origin_y, difficulty, cfg, platform_size=2.0):
    horizontal_scale = float(cfg.horizontal_scale)
    vertical_scale = float(cfg.vertical_scale)
    env_length = float(cfg.terrain_length)
    env_width = float(cfg.terrain_width)

    beam_length = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
    beam_gap = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
    step_height = 0.04 if difficulty < 0.2 else 0.05 + 0.18 * difficulty
    max_height = int(step_height / vertical_scale) / 100.0
    height_candidates = np.arange(0.0, max(max_height, 0.04), 0.04)
    if height_candidates.size == 0:
        height_candidates = np.array([0.0], dtype=np.float32)

    boxes = []
    x_cursor = float(platform_size)
    while x_cursor < env_length:
        box_length = min(beam_length, env_length - x_cursor)
        if box_length <= 0.0:
            break
        beam_width = 2 * random.randint(7, 15) * horizontal_scale
        beam_center_z = float(np.random.choice(height_candidates))
        size = np.array([box_length, beam_width, 0.1], dtype=np.float32)
        center = np.array(
            [
                origin_x + x_cursor + box_length / 2,
                origin_y + env_width / 2,
                beam_center_z,
            ],
            dtype=np.float32,
        )
        boxes.append((center, size))
        x_cursor += beam_length + beam_gap
    return boxes


def _build_air_stone_boxes(origin_x, origin_y, difficulty, cfg):
    env_length = float(cfg.terrain_length)
    env_width = float(cfg.terrain_width)
    difficulty = _normalized_difficulty(difficulty, cfg)
    x_shift = float(getattr(cfg, "mgdp_air_stone_x_shift", -0.9))

    box_size_x = 4.5
    box_size_y = np.random.randint(18, 25) / 10
    box_size_z = 0.5
    clearance_height = 0.5 - (0.5 - 0.25) * difficulty

    center = np.array(
        [
            origin_x + env_length * 0.5 + x_shift,
            origin_y + env_width / 2,
            clearance_height + box_size_z / 2,
        ],
        dtype=np.float32,
    )
    size = np.array([box_size_x, box_size_y, box_size_z], dtype=np.float32)
    return [(center, size)]


def append_mgdp_extra_trimeshes(height_field_raw, row, col, extra_specs, cfg, spacing=0.0):
    if not extra_specs:
        return [], []

    origin_x = float(cfg.border_size) + row * (float(cfg.terrain_length) + float(spacing))
    origin_y = float(cfg.border_size) + col * (float(cfg.terrain_width) + float(spacing))
    horizontal_scale = float(cfg.horizontal_scale)
    vertical_scale = float(cfg.vertical_scale)

    vertices, triangles = [], []
    for spec in extra_specs:
        spec_type = spec.get("type", "")
        if spec_type == "air_beams":
            if not bool(getattr(cfg, "add_air_beam", True)):
                continue
            boxes = _build_air_beam_boxes(
                origin_x,
                origin_y,
                float(spec.get("difficulty", 0.0)),
                cfg,
                platform_size=float(spec.get("platform_size", 2.0)),
            )
        elif spec_type == "air_stones":
            if not bool(getattr(cfg, "add_air_stone", True)):
                continue
            boxes = _build_air_stone_boxes(
                origin_x,
                origin_y,
                float(spec.get("difficulty", 0.0)),
                cfg,
            )
        else:
            continue

        for center, size in boxes:
            mesh_vertices, mesh_triangles = _box_trimesh(size, center)
            vertices.append(mesh_vertices)
            triangles.append(mesh_triangles)
            _stamp_box_height(
                height_field_raw,
                center,
                size,
                horizontal_scale,
                vertical_scale,
            )

    return vertices, triangles
