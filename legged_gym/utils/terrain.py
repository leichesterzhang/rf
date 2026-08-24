from collections import defaultdict
import numpy as np
from numpy.random import choice
from scipy import interpolate
from scipy.ndimage import binary_dilation

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain_mgdp import append_mgdp_extra_trimeshes, build_mgdp_parkour_terrain


def convert_heightfield_to_trimesh_with_x_edges(
    height_field_raw,
    horizontal_scale,
    vertical_scale,
    slope_threshold=None,
):
    hf = height_field_raw
    num_rows, num_cols = hf.shape

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    move_x = np.zeros((num_rows, num_cols))
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))

        move_x[: num_rows - 1, :] += (hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, : num_cols - 1] += (hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )

        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.spacing = cfg.terrain_spacing
        self.spacing_pixels = int(self.spacing / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels + max(0, cfg.num_cols-1) * self.spacing_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels + max(0, cfg.num_rows-1) * self.spacing_pixels) + 2 * self.border
        self.name2cols = defaultdict(set)  # terrain type to column index
        self.cols2id = []  # column index to terrain id
        self.terrain_id_map = -np.ones((cfg.num_rows, cfg.num_cols), dtype=np.int64)
        self.spawn_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        self._mgdp_extra_trimesh_specs = []

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        self.x_edge_mask = None
        if self.type in ["heightfield", "trimesh"]:
            vertices, triangles, x_edge_mask = convert_heightfield_to_trimesh_with_x_edges(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
            edge_width_thresh = getattr(self.cfg, "edge_width_thresh", None)
            if edge_width_thresh is not None and edge_width_thresh > 0.0:
                half_edge_width = int(edge_width_thresh / self.cfg.horizontal_scale)
                if half_edge_width > 0:
                    structure = np.ones((half_edge_width * 2 + 1, 1), dtype=bool)
                    x_edge_mask = binary_dilation(x_edge_mask, structure=structure)
            self.x_edge_mask = x_edge_mask
            if self.type == "trimesh":
                self.vertices, self.triangles = vertices, triangles
                if getattr(self.cfg, "terrain_style", "") == "mgdp_parkour":
                    self._append_mgdp_extra_trimeshes()
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            if getattr(self.cfg, "terrain_style", "") == "mgdp_parkour":
                difficulty = np.random.choice([0.9, 0.95])
            else:
                difficulty = np.random.uniform(0.9, 0.95)

            terrain = self.make_terrain(choice, difficulty)
            self.terrain_id_map[i, j] = getattr(terrain, "terrain_id", -1)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.terrain_id_map[i, j] = getattr(terrain, "terrain_id", -1)
                self.add_terrain_to_map(terrain, i, j)
            self.name2cols[terrain.terrain_name].add(j)
            self.cols2id.append(terrain.terrain_id)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.length_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if getattr(self.cfg, "terrain_style", "") == "mgdp_parkour":
            return build_mgdp_parkour_terrain(terrain, choice, difficulty, self.cfg, self.proportions)
        has_pit_terrain = len(self.proportions) >= 10
        IS_HARD = True
        if IS_HARD:
            # hard
            slope = 0.1 + difficulty * 0.52  # max: 29.6 degrees
            step_height = 0.05 + 0.23 * difficulty  # max: 0.257 m
            discrete_obstacles_height = 0.05 + difficulty * 0.25  # max: 0.275 m
        else:
            # default (easy)
            slope = difficulty * 0.4  # max: 19.8 degrees
            step_height = 0.05 + 0.18 * difficulty  # max: 0.212 m
            discrete_obstacles_height = 0.05 + difficulty * 0.2  # max: 0.23 m

        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 0.8 * difficulty
        pit_depth_range = getattr(self.cfg, "pit_depth_range", [0.15, 0.35])
        pit_platform_size = getattr(self.cfg, "pit_platform_size", 1.0)
        pit_depth = pit_depth_range[0] + difficulty * (pit_depth_range[1] - pit_depth_range[0])
        amplitude = 0.1 + 0.2 * difficulty
        
        if choice < self.proportions[0]:
            terrain.terrain_name = "wave"
            terrain.terrain_id = 0
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[1]:  # 平滑坡
            terrain.terrain_name = "slope"
            terrain.terrain_id = 1
            if choice < (self.proportions[0] + self.proportions[1])/ 2:  # 一半正坡, 一半负坡
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[2]:  # 粗糙坡
            terrain.terrain_name = "rough_slope"
            terrain.terrain_id = 2
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:  # 下楼梯
            terrain.terrain_name = "stairs_down"
            terrain.terrain_id = 4
            if choice<self.proportions[3]:  # 上楼梯
                terrain.terrain_name = "stairs_up"
                terrain.terrain_id = 3
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[5]:  # 障碍物
            terrain.terrain_name = "obstacles"
            terrain.terrain_id = 5
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[6]:  # 梅花桩
            terrain.terrain_name = "stepping_stones"
            terrain.terrain_id = 6
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[7]:  # 间隙
            terrain.terrain_name = "gap"
            terrain.terrain_id = 7
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif has_pit_terrain and choice < self.proportions[8]:  # 中心坑
            terrain.terrain_name = "pit"
            terrain.terrain_id = 8
            pit_terrain(terrain, depth=pit_depth, platform_size=pit_platform_size)
        else:  # 平地
            terrain.terrain_name = "flat"
            terrain.terrain_id = 9 if has_pit_terrain else 8
            pit_terrain(terrain, depth=0.0, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * (self.length_per_env_pixels + self.spacing_pixels)
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + j * (self.width_per_env_pixels + self.spacing_pixels)
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length + i * self.spacing
        env_origin_y = (j + 0.5) * self.env_width + j * self.spacing
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.spawn_origins[i, j] = self._compute_spawn_origin(terrain, i, j)
        extra_specs = getattr(terrain, "extra_trimesh_specs", None)
        if extra_specs:
            self._mgdp_extra_trimesh_specs.append((i, j, list(extra_specs)))

    def _append_extra_trimeshes(self, extra_vertices, extra_triangles):
        if not extra_vertices:
            return
        vertices = [self.vertices]
        triangles = [self.triangles]
        vertex_offset = self.vertices.shape[0]
        for mesh_vertices, mesh_triangles in zip(extra_vertices, extra_triangles):
            mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32)
            mesh_triangles = np.asarray(mesh_triangles, dtype=np.uint32)
            vertices.append(mesh_vertices)
            triangles.append(mesh_triangles + vertex_offset)
            vertex_offset += mesh_vertices.shape[0]
        self.vertices = np.concatenate(vertices, axis=0)
        self.triangles = np.concatenate(triangles, axis=0)

    def _append_mgdp_extra_trimeshes(self):
        extra_vertices = []
        extra_triangles = []
        for row, col, extra_specs in self._mgdp_extra_trimesh_specs:
            mesh_vertices, mesh_triangles = append_mgdp_extra_trimeshes(
                self.height_field_raw,
                row,
                col,
                extra_specs,
                self.cfg,
                spacing=self.spacing,
            )
            extra_vertices.extend(mesh_vertices)
            extra_triangles.extend(mesh_triangles)
        self._append_extra_trimeshes(extra_vertices, extra_triangles)

    def _compute_spawn_origin(self, terrain, row, col):
        spawn_region = getattr(terrain, "spawn_region_px", None)
        if spawn_region is None:
            spawn_length = float(getattr(self.cfg, "start_platform_length", 2.0))
            spawn_width = float(getattr(self.cfg, "start_platform_width", min(2.0, self.env_width)))
            spawn_x1 = 0
            spawn_x2 = max(1, min(terrain.width, int(spawn_length / terrain.horizontal_scale)))
            center_y = terrain.length // 2
            half_width = max(1, int(0.5 * spawn_width / terrain.horizontal_scale))
            spawn_y1 = max(0, center_y - half_width)
            spawn_y2 = min(terrain.length, center_y + half_width)
        else:
            spawn_x1, spawn_x2, spawn_y1, spawn_y2 = [int(v) for v in spawn_region]
            spawn_x1 = max(0, min(spawn_x1, terrain.width - 1))
            spawn_x2 = max(spawn_x1 + 1, min(spawn_x2, terrain.width))
            spawn_y1 = max(0, min(spawn_y1, terrain.length - 1))
            spawn_y2 = max(spawn_y1 + 1, min(spawn_y2, terrain.length))

        local_x = 0.5 * (spawn_x1 + spawn_x2) * terrain.horizontal_scale
        local_y = 0.5 * (spawn_y1 + spawn_y2) * terrain.horizontal_scale
        local_z = np.max(terrain.height_field_raw[spawn_x1:spawn_x2, spawn_y1:spawn_y2]) * terrain.vertical_scale
        world_x = row * (self.env_length + self.spacing) + local_x
        world_y = col * (self.env_width + self.spacing) + local_y
        return np.array([world_x, world_y, local_z], dtype=np.float32)

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
