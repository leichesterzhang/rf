import copy
import os
import xml.etree.ElementTree as ET


SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Z1_NOLIDARV1.1.SLDASM",
    "urdf",
    "Z1_NOLIDARV1.1.SLDASM.urdf",
)
DST = os.path.join(
    os.path.dirname(__file__),
    "..",
    "RM75",
    "urdf",
    "Z1_NOLIDARV11_locked_arm.urdf",
)


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.attrib["name"] = "Z1_NOLIDARV1.1_locked_arm"

    # The current RM75 policy controls only the 12 leg joints. Keep all arm
    # mass/inertia in the rigid-body tree, but lock the seven arm joints.
    arm_joints = {
        "link1_joint",
        "link2_joint",
        "link3_joint",
        "link4_joint",
        "link5_joint",
        "link6_joint",
        "link7_joint",
    }
    for joint in root.findall("joint"):
        if joint.attrib.get("name") in arm_joints:
            joint.attrib["type"] = "fixed"
            for tag in ("axis", "limit", "dynamics", "safety_controller"):
                child = joint.find(tag)
                if child is not None:
                    joint.remove(child)

    # Move, rather than duplicate, the exported toe collision sphere. The
    # training environment needs a dedicated rigid body for each foot so its
    # contact force and slip velocity can be measured independently.
    for prefix in ("FL", "FR", "RL", "RR"):
        calf = root.find(f"link[@name='{prefix}_calf']")
        if calf is None:
            raise RuntimeError(f"Missing {prefix}_calf in exported URDF")
        for collision in list(calf.findall("collision")):
            if collision.attrib.get("name") == f"{prefix}_foot_collision":
                calf.remove(collision)

    # Preserve the four-foot interface expected by RM75VisualRampTrotRobot.
    for prefix in ("FL", "FR", "RL", "RR"):
        link = ET.Element("link", {"name": f"{prefix}_foot"})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        # Keep this contact body dynamically well-conditioned.  With an almost
        # massless body (1e-6 kg / 1e-12 kg m^2), PhysX reports the ground
        # impulse on the parent calf instead of on this named foot.  That makes
        # every foot-contact gait reward read as airborne.  The legacy RM75
        # model used these values successfully; the added 0.16 kg total is
        # negligible relative to the 98 kg robot and is covered by mass DR.
        ET.SubElement(inertial, "mass", {"value": "0.04"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": "9.6e-06",
                "ixy": "0",
                "ixz": "0",
                "iyy": "9.6e-06",
                "iyz": "0",
                "izz": "9.6e-06",
            },
        )
        collision = ET.SubElement(link, "collision", {"name": f"{prefix}_foot_collision"})
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": "0.032"})
        root.append(link)

        joint = ET.Element("joint", {"name": f"{prefix}_foot_joint", "type": "fixed", "dont_collapse": "true"})
        ET.SubElement(joint, "origin", {"xyz": "0 0 -0.35", "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": f"{prefix}_calf"})
        ET.SubElement(joint, "child", {"link": f"{prefix}_foot"})
        root.append(joint)

    # Isaac Gym resolves relative paths reliably; ROS package:// paths are not
    # assumed to be available in the training container.
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("package://Z1_NOLIDARV1.1.SLDASM/"):
            mesh.attrib["filename"] = "../meshes/" + filename.split("/meshes/", 1)[1]

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print(DST)


if __name__ == "__main__":
    main()
