#!/usr/bin/env python3
"""Generate fixed-arm RM75 compatibility and MuJoCo resources."""

import math
import re
import shutil
from copy import deepcopy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF_CANDIDATES = (
    ROOT / "RM75" / "urdf" / "Z1_NOLIDAR.urdf",
    ROOT / "RM75" / "urdf" / "\u56db\u8db3\u673a\u5668\u4ebaRM75.urdf",
)
SOURCE_URDF = next(
    (path for path in SOURCE_URDF_CANDIDATES if path.is_file()),
    SOURCE_URDF_CANDIDATES[0],
)
LOCKED_URDF = ROOT / "RM75" / "urdf" / "RM75_locked_arm.urdf"
MUJOCO_DIR = ROOT / "resources" / "robots" / "RM75"
MUJOCO_MODEL = MUJOCO_DIR / "RM75.xml"
DOG_502_SCENE_DIR = ROOT / "resources" / "robots" / "dog_502"
# Keep the zero-action stance close to the flat-ground contact height.  Starting
# higher makes the heavy RM75 drop onto the front feet before the rear feet.
DEFAULT_BASE_Z = 0.53
ROS_MESH_PREFIX = "package://Z1_NOLIDAR/meshes/"
LOCAL_MESH_PREFIX = "../meshes/"

ARM_FIXED_ANGLES = {
    "link1_joint": 0.0,
    "link2_joint": 0.0,
    "link3_joint": 0.0,
    "link4_joint": 0.0,
    "link5_joint": 0.0,
    "link6_joint": 0.0,
    "link7_joint": 0.0,
}

LEG_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

STABLE_LEG_ZERO_OFFSETS = (
    0.06,
    0.70,
    -1.20,
    -0.06,
    0.70,
    -1.20,
    0.06,
    0.75,
    -1.175,
    -0.06,
    0.75,
    -1.175,
)
LEG_ZERO_OFFSETS = dict(zip(LEG_JOINT_ORDER, STABLE_LEG_ZERO_OFFSETS))
RAW_LEG_LIMITS = {
    joint_name: (
        (-0.87, 0.87)
        if "_hip_" in joint_name
        else (-0.94, 4.69)
        if "_thigh_" in joint_name
        else (-2.82, -0.43)
    )
    for joint_name in LEG_JOINT_ORDER
}

DEFAULT_LEG_ANGLES = tuple(0.0 for _ in LEG_JOINT_ORDER)

SCENE_NAMES = (
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


def format_values(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def parse_values(element, attribute, default):
    value = element.get(attribute)
    if value is None:
        return list(default)
    return [float(item) for item in value.split()]


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def matmul(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def axis_angle_matrix(axis, angle):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return (
        (
            x * x * one_minus_c + c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ),
        (
            y * x * one_minus_c + z * s,
            y * y * one_minus_c + c,
            y * z * one_minus_c - x * s,
        ),
        (
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            z * z * one_minus_c + c,
        ),
    )


def matrix_to_quat_wxyz(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    return (qw, qx, qy, qz)


def matrix_to_rpy_xyz(matrix):
    pitch = math.atan2(
        -matrix[2][0],
        math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0]),
    )
    if abs(abs(pitch) - math.pi / 2.0) < 1.0e-9:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    else:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    return (roll, pitch, yaw)


def indent_xml(element, level=0):
    prefix = "\n" + "  " * level
    child_prefix = "\n" + "  " * (level + 1)
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in children:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_prefix
        children[-1].tail = prefix
    if level == 0:
        element.tail = "\n"


def write_xml(root, destination):
    indent_xml(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def apply_joint_zero_offset(joint, angle):
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    origin_rpy = parse_values(origin, "rpy", (0.0, 0.0, 0.0))
    axis = parse_values(joint.find("axis"), "xyz", (0.0, 0.0, 1.0))
    rotation = matmul(rpy_matrix(*origin_rpy), axis_angle_matrix(axis, angle))
    origin.set("rpy", format_values(matrix_to_rpy_xyz(rotation)))


def matrices_close(left, right, tol=1.0e-8):
    return all(
        abs(left[row][col] - right[row][col]) <= tol
        for row in range(3)
        for col in range(3)
    )


def joint_has_zero_offset(joint, angle):
    origin = joint.find("origin")
    axis = joint.find("axis")
    if origin is None or axis is None:
        return abs(angle) < 1.0e-8
    origin_rpy = parse_values(origin, "rpy", (0.0, 0.0, 0.0))
    joint_axis = parse_values(axis, "xyz", (0.0, 0.0, 1.0))
    return matrices_close(rpy_matrix(*origin_rpy), axis_angle_matrix(joint_axis, angle))


def infer_leg_zero_offset(joint):
    limit = joint.find("limit")
    if limit is None:
        return 0.0
    raw_lower, _ = RAW_LEG_LIMITS[joint.get("name")]
    return raw_lower - float(limit.get("lower"))


def ensure_leg_zero_offsets(root):
    changed = False
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name not in LEG_ZERO_OFFSETS:
            continue
        angle = LEG_ZERO_OFFSETS[joint_name]
        current_angle = infer_leg_zero_offset(joint)
        delta_angle = angle - current_angle
        if abs(delta_angle) < 1.0e-9 and joint_has_zero_offset(joint, angle):
            continue
        apply_joint_zero_offset(joint, delta_angle)
        limit = joint.find("limit")
        if limit is not None:
            raw_lower, raw_upper = RAW_LEG_LIMITS[joint_name]
            lower = raw_lower - angle
            upper = raw_upper - angle
            limit.set("lower", f"{lower:.12g}")
            limit.set("upper", f"{upper:.12g}")
        changed = True
    return changed


def ensure_fixed_arm_pose(root):
    changed = False
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name not in ARM_FIXED_ANGLES:
            continue
        angle = ARM_FIXED_ANGLES[joint_name]
        if joint.get("type") != "fixed":
            apply_joint_zero_offset(joint, angle)
            joint.set("type", "fixed")
            changed = True
        elif abs(angle) > 1.0e-9:
            raise ValueError(
                f"Cannot bake nonzero angle {angle} into already fixed {joint_name}"
            )
        for tag in ("axis", "limit", "dynamics", "safety_controller", "calibration"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
                changed = True
    return changed


def ensure_foot_links(root):
    """Move the terminal foot spheres out of calf links.

    Isaac Gym identifies feet by rigid-body name.  Keeping the sphere inside
    ``*_calf`` makes the whole lower leg a foot and lets calf contact escape
    the collision penalty, which encourages a low crawling posture.
    """
    changed = False
    for leg in ("FL", "FR", "RL", "RR"):
        calf_name = f"{leg}_calf"
        foot_name = f"{leg}_foot"
        foot_joint_name = f"{foot_name}_joint"
        calf = root.find(f"link[@name='{calf_name}']")
        if calf is None:
            raise ValueError(f"Missing {calf_name} link")

        existing_foot = root.find(f"link[@name='{foot_name}']")
        existing_joint = root.find(f"joint[@name='{foot_joint_name}']")
        if existing_foot is not None or existing_joint is not None:
            if existing_foot is None or existing_joint is None:
                raise ValueError(f"{foot_name} and {foot_joint_name} must be defined together")
            continue

        foot_collision = None
        for collision in calf.findall("collision"):
            geometry = collision.find("geometry")
            if (
                collision.get("name") == f"{foot_name}_collision"
                or (
                    geometry is not None
                    and geometry.find("sphere") is not None
                    and parse_values(collision.find("origin"), "xyz", (0.0, 0.0, 0.0))[2] < -0.30
                )
            ):
                foot_collision = collision
                break
        if foot_collision is None:
            raise ValueError(f"Missing terminal foot collision in {calf_name}")

        collision_origin = foot_collision.find("origin")
        joint_xyz = parse_values(collision_origin, "xyz", (0.0, 0.0, 0.0))
        joint_rpy = parse_values(collision_origin, "rpy", (0.0, 0.0, 0.0))
        geometry = deepcopy(foot_collision.find("geometry"))
        calf.remove(foot_collision)

        foot = ET.Element("link", name=foot_name)
        inertial = ET.SubElement(foot, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="0.04")
        ET.SubElement(
            inertial,
            "inertia",
            ixx="9.6e-06",
            ixy="0",
            ixz="0",
            iyy="9.6e-06",
            iyz="0",
            izz="9.6e-06",
        )
        collision = ET.SubElement(foot, "collision", name=f"{foot_name}_collision")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        collision.append(geometry)
        root.append(foot)

        joint = ET.Element("joint", name=foot_joint_name, type="fixed", dont_collapse="true")
        ET.SubElement(
            joint,
            "origin",
            xyz=format_values(joint_xyz),
            rpy=format_values(joint_rpy),
        )
        ET.SubElement(joint, "parent", link=calf_name)
        ET.SubElement(joint, "child", link=foot_name)
        root.append(joint)
        changed = True

    return changed


def ensure_local_mesh_paths(root):
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(ROS_MESH_PREFIX):
            mesh.set("filename", LOCAL_MESH_PREFIX + Path(filename).name)
            changed = True
    return changed


def validate_training_urdf(root, urdf_path):
    moving_joint_names = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    )
    if moving_joint_names != LEG_JOINT_ORDER:
        raise ValueError(
            f"Expected exactly 12 leg DOFs, found {moving_joint_names}"
        )

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}
    for index in range(1, 7):
        joint_name = f"link{index}_joint"
        link_name = f"link{index}"
        if joints[joint_name].get("type") != "fixed":
            raise ValueError(f"{joint_name} must be fixed")
        for element_path in ("inertial/mass", "visual/geometry", "collision/geometry"):
                if links[link_name].find(element_path) is None:
                    raise ValueError(f"{link_name} is missing {element_path}")

    for leg in ("FL", "FR", "RL", "RR"):
        foot_link = links.get(f"{leg}_foot")
        foot_joint = joints.get(f"{leg}_foot_joint")
        if foot_link is None or foot_joint is None:
            raise ValueError(f"Missing fixed foot link/joint for {leg}")
        if foot_joint.get("type") != "fixed":
            raise ValueError(f"{leg}_foot_joint must be fixed")
        if foot_link.find("inertial/mass") is None:
            raise ValueError(f"{leg}_foot is missing inertial mass")
        if foot_link.find("collision/geometry") is None:
            raise ValueError(f"{leg}_foot is missing collision geometry")

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://"):
            raise ValueError(f"Isaac Gym cannot resolve ROS mesh URI {filename}")
        mesh_path = (urdf_path.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing URDF mesh {mesh_path}")


def build_training_urdf():
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    changed = ensure_leg_zero_offsets(root)
    changed = ensure_fixed_arm_pose(root) or changed
    changed = ensure_foot_links(root) or changed
    changed = ensure_local_mesh_paths(root) or changed
    validate_training_urdf(root, SOURCE_URDF)
    if changed:
        write_xml(root, SOURCE_URDF)
    return changed


def build_locked_urdf():
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    ensure_leg_zero_offsets(root)
    ensure_fixed_arm_pose(root)
    ensure_foot_links(root)
    ensure_local_mesh_paths(root)
    validate_training_urdf(root, LOCKED_URDF)
    root.set("name", "RM75_locked_arm")
    write_xml(root, LOCKED_URDF)


def read_robot_description(source_urdf):
    root = ET.parse(source_urdf).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    child_to_joint = {}
    children_by_parent = {}
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        child_to_joint[child] = joint
        children_by_parent.setdefault(parent, []).append(child)
    return links, child_to_joint, children_by_parent


def add_inertial(body, link):
    inertial = link.find("inertial")
    if inertial is None:
        return

    origin = inertial.find("origin")
    mass = float(inertial.find("mass").get("value"))
    inertia = inertial.find("inertia")
    position = parse_values(origin, "xyz", (0.0, 0.0, 0.0))
    full_inertia = (
        float(inertia.get("ixx")),
        float(inertia.get("iyy")),
        float(inertia.get("izz")),
        float(inertia.get("ixy")),
        float(inertia.get("ixz")),
        float(inertia.get("iyz")),
    )
    if mass <= 0.0:
        mass = 1.0e-6
        full_inertia = (1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0)
    ET.SubElement(
        body,
        "inertial",
        pos=format_values(position),
        mass=f"{mass:.12g}",
        fullinertia=format_values(full_inertia),
    )


def mesh_filename(link):
    visual_mesh = link.find("visual/geometry/mesh")
    if visual_mesh is None:
        return None
    return Path(visual_mesh.get("filename")).name


def add_visual_geometry(body, link, link_name):
    mesh_name = f"{link_name}_mesh"
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{link_name}_visual",
            "type": "mesh",
            "mesh": mesh_name,
            "class": "visual",
            "material": "rm75_grey",
        },
    )


def add_collision_geometries(body, link, link_name):
    collisions = link.findall("collision")
    for index, collision in enumerate(collisions):
        geometry = collision.find("geometry")
        if geometry is None:
            continue

        origin = collision.find("origin")
        pos = parse_values(origin, "xyz", (0.0, 0.0, 0.0))
        rpy = parse_values(origin, "rpy", (0.0, 0.0, 0.0))
        attributes = {
            "name": f"{link_name}_collision_{index}",
            "pos": format_values(pos),
            "quat": format_values(matrix_to_quat_wxyz(rpy_matrix(*rpy))),
            "group": "3",
        }

        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        sphere = geometry.find("sphere")
        mesh = geometry.find("mesh")
        if box is not None:
            size = [0.5 * value for value in parse_values(box, "size", (0.0, 0.0, 0.0))]
            attributes.update(type="box", size=format_values(size))
        elif cylinder is not None:
            radius = float(cylinder.get("radius"))
            half_length = 0.5 * float(cylinder.get("length"))
            attributes.update(type="cylinder", size=format_values((radius, half_length)))
        elif sphere is not None:
            attributes.update(type="sphere", size=str(float(sphere.get("radius"))))
        elif mesh is not None:
            attributes.update(type="mesh", mesh=f"{link_name}_mesh")
        else:
            continue

        ET.SubElement(body, "geom", **attributes)


def build_mujoco_model():
    links, child_to_joint, children_by_parent = read_robot_description(LOCKED_URDF)
    model = ET.Element("mujoco", model="RM75_locked_arm")
    ET.SubElement(
        model,
        "compiler",
        angle="radian",
        meshdir="../../../RM75/meshes",
        convexhull="false",
    )

    default = ET.SubElement(model, "default")
    ET.SubElement(default, "joint", damping="0.05", armature="0.02", frictionloss="0.02")
    ET.SubElement(default, "geom", condim="3", friction="0.9 0.02 0.01")
    visual_default = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(
        visual_default,
        "geom",
        type="mesh",
        contype="0",
        conaffinity="0",
        group="2",
    )

    asset = ET.SubElement(model, "asset")
    ET.SubElement(asset, "material", name="rm75_grey", rgba="0.70 0.70 0.70 1")
    ET.SubElement(asset, "material", name="rm75_white", rgba="0.92 0.92 0.92 1")
    for link_name, link in links.items():
        filename = mesh_filename(link)
        if filename:
            ET.SubElement(asset, "mesh", name=f"{link_name}_mesh", file=filename)

    worldbody = ET.SubElement(model, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base", pos=f"0 0 {DEFAULT_BASE_Z:.12g}")
    ET.SubElement(base, "freejoint", name="root")

    def append_link(body, link_name):
        link = links[link_name]
        add_inertial(body, link)
        if mesh_filename(link):
            add_visual_geometry(body, link, link_name)
        add_collision_geometries(body, link, link_name)
        for child_name in children_by_parent.get(link_name, []):
            joint = child_to_joint[child_name]
            origin = joint.find("origin")
            origin_xyz = parse_values(origin, "xyz", (0.0, 0.0, 0.0))
            origin_rpy = parse_values(origin, "rpy", (0.0, 0.0, 0.0))
            rotation = rpy_matrix(*origin_rpy)
            joint_name = joint.get("name")

            child_body = ET.SubElement(
                body,
                "body",
                name=child_name,
                pos=format_values(origin_xyz),
                quat=format_values(matrix_to_quat_wxyz(rotation)),
            )
            if joint.get("type") != "fixed":
                limit = joint.find("limit")
                joint_attributes = {
                    "name": joint_name,
                    "axis": joint.find("axis").get("xyz"),
                }
                if limit is not None:
                    joint_attributes["range"] = (
                        f"{limit.get('lower')} {limit.get('upper')}"
                    )
                ET.SubElement(child_body, "joint", **joint_attributes)
            append_link(child_body, child_name)

        if link_name.endswith("_foot"):
            ET.SubElement(
                body,
                "geom",
                name=f"{link_name}_contact_visual",
                type="sphere",
                size="0.032",
                material="rm75_white",
                contype="0",
                conaffinity="0",
                group="2",
            )

    append_link(base, "base_link")

    actuator = ET.SubElement(model, "actuator")
    for joint_name in LEG_JOINT_ORDER:
        effort = float(child_to_joint[next(
            child for child, joint in child_to_joint.items() if joint.get("name") == joint_name
        )].find("limit").get("effort"))
        ET.SubElement(
            actuator,
            "motor",
            name=f"{joint_name}_motor",
            joint=joint_name,
            ctrllimited="true",
            ctrlrange=f"{-effort:.12g} {effort:.12g}",
        )

    write_xml(model, MUJOCO_MODEL)


def build_scenes():
    MUJOCO_DIR.mkdir(parents=True, exist_ok=True)
    source_images = DOG_502_SCENE_DIR / "imgs"
    if source_images.is_dir():
        shutil.copytree(source_images, MUJOCO_DIR / "imgs", dirs_exist_ok=True)
    keyframe_qpos = format_values((0.0, 0.0, DEFAULT_BASE_Z, 1.0, 0.0, 0.0, 0.0, *DEFAULT_LEG_ANGLES))
    for scene_name in SCENE_NAMES:
        source = DOG_502_SCENE_DIR / f"{scene_name}.xml"
        destination = MUJOCO_DIR / f"{scene_name}.xml"
        text = source.read_text(encoding="utf-8")
        text = text.replace('file="dog_502.xml"', 'file="RM75.xml"')
        text = text.replace("dog_502", "RM75")
        text = re.sub(r'qpos="[^"]+"', f'qpos="{keyframe_qpos}"', text)
        if "<keyframe" not in text:
            keyframe = (
                "\n  <keyframe>\n"
                f'    <key name="{scene_name}_start" qpos="{keyframe_qpos}"/>\n'
                "  </keyframe>\n"
            )
            text = text.replace("</mujoco>", f"{keyframe}</mujoco>")
        destination.write_text(text, encoding="utf-8")


def main():
    source_changed = build_training_urdf()
    build_locked_urdf()
    build_mujoco_model()
    build_scenes()
    print(f"Using source {SOURCE_URDF.relative_to(ROOT)}")
    if source_changed:
        print(
            f"Updated {SOURCE_URDF.relative_to(ROOT)} with stable leg zero offsets "
            "and a fixed arm"
        )
    else:
        print(
            f"Verified 12 leg DOFs and fixed arm in "
            f"{SOURCE_URDF.relative_to(ROOT)}"
        )
    print(f"Generated {LOCKED_URDF.relative_to(ROOT)}")
    print(f"Generated {MUJOCO_MODEL.relative_to(ROOT)} and {len(SCENE_NAMES)} terrain scenes")


if __name__ == "__main__":
    main()
