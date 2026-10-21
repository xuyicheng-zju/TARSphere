"""Compose Dex3-1 hands onto the locomotion-certified Unitree G1 asset.

The composed layer remounts donor hands and body colliders, then overlays
visual OmniPBR Looks. Joint physics stay on the certified body.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import uuid

from tarsphere.embodiment.dex3 import (
    DEX3_DYNAMIC_FRICTION,
    DEX3_JOINT_NAMES,
    DEX3_LINK_NAMES,
    DEX3_MINIMUM_DIAGONAL_INERTIA_KG_M2,
    DEX3_MINIMUM_LINK_MASS_KG,
    DEX3_STATIC_FRICTION,
)


G1_STABLE_ROOT_PRIM = "/g1_29dof"
G1_DEX3_DONOR_ROOT_PRIM = "/g1_29dof_with_hand_rev_1_0"

# Locomotion-certified ``g1_29dof_color.usd`` only authors ankle Mesh colliders.
# Without the donor body collision payloads the visual torso/legs/arms pass
# through furniture while PhysX and ``physical_contact`` stay quiet. These
# shared rigid links carry enabled CollisionAPI geometry on the Unitree Dex3
# donor and must be remounted onto the stable body.
G1_BODY_COLLISION_LINK_NAMES: tuple[str, ...] = (
    "pelvis_contour_link",
    "torso_link",
    "head_link",
    "logo_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

# Foot-only Mesh prims that ship on the loco USD. Disable them after the donor
# ankle sphere hulls are mounted so the body does not carry duplicate feet.
G1_LOCO_FOOT_COLLISION_LINKS: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

# Donor ``g1_29dof_with_hand_rev_1_0.usd`` authors waist pitch as revolute
# with stiffness 625, effort 35 N·m, and +/-29.79 deg on axis Y. Damping 0
# does not hold torso pitch under load; 10 matches waist-yaw policy damping
# scaled for the stiffer pitch drive.
G1_WAIST_PITCH_DRIVE_STIFFNESS = 625.0
G1_WAIST_PITCH_DRIVE_DAMPING = 10.0
G1_WAIST_PITCH_DRIVE_MAX_FORCE = 35.0
G1_WAIST_PITCH_LOWER_LIMIT_DEG = -29.79380226135254
G1_WAIST_PITCH_UPPER_LIMIT_DEG = 29.79380226135254

# Same donor drive as pitch. Applied only by the isolated VLA overlay; ordinary
# grasp/loco/oracle keep waist_roll as the locomotion-certified FixedJoint.
G1_WAIST_ROLL_DRIVE_STIFFNESS = 625.0
G1_WAIST_ROLL_DRIVE_DAMPING = 10.0
G1_WAIST_ROLL_DRIVE_MAX_FORCE = 35.0
G1_WAIST_ROLL_LOWER_LIMIT_DEG = -29.79380226135254
G1_WAIST_ROLL_UPPER_LIMIT_DEG = 29.79380226135254

# The locomotion-certified body is the original G1 geometry.  The Dex3 donor
# is rev-1.0 and moved the same waist anchors to 44 mm + 0 mm.  Referencing a
# donor PhysicsJoint without overriding localPos0 mixes the two revisions:
# the stable torso/arms then sit 19 mm low on the ordinary composition and
# 10 mm low when the VLA interface unlocks roll. Inherit joint type, limits, axis and
# drive schema from the donor, but keep the stable body's physical anchors.
G1_STABLE_WAIST_ROLL_LOCAL_POS0 = (-0.0039635, 0.0, 0.035)
G1_STABLE_WAIST_PITCH_LOCAL_POS0 = (0.0, 0.0, 0.019)

# Official Unitree / TARSphere G1 ships constant OmniPBR dark≈0.2 / white≈0.7
# with no albedo textures. These values retune that split toward the real robot
# on the composed Looks only: darker chassis, warmer off-white limbs.
G1_LOOK_DARK_DIFFUSE = (0.08, 0.08, 0.08)
G1_LOOK_WHITE_DIFFUSE = (0.82, 0.80, 0.76)
G1_LOOK_DARK_ROUGHNESS = 0.35
G1_LOOK_WHITE_ROUGHNESS = 0.40
G1_LOOK_DARK_METALLIC = 0.15
G1_LOOK_WHITE_METALLIC = 0.0
# Dex3 donor Looks sit outside the referenced hand subtree, so the 16 links
# need a composed-root material. Real Dex3 housings are dark rubber, not the
# donor's unresolved white plastic fallback.
G1_LOOK_DEX3_NAME = "dex3"
G1_LOOK_DEX3_DIFFUSE = G1_LOOK_DARK_DIFFUSE
G1_LOOK_DEX3_ROUGHNESS = 0.45
G1_LOOK_DEX3_METALLIC = 0.05
G1_LOOK_DEX3_PRIM = f"{G1_STABLE_ROOT_PRIM}/Looks/{G1_LOOK_DEX3_NAME}"


def create_g1_with_gravity_neutral_dex3(
    stable_g1_usd: Path | str,
    dex3_donor_usd: Path | str,
) -> Path:
    """Mount gravity-neutral donor Dex3 hands on the stable G1 body."""
    return _write_composed_g1(
        stable_g1_usd,
        dex3_donor_usd,
        unlock_waist_roll=False,
    )


def create_g1_with_unlocked_waist_roll(
    stable_g1_usd: Path | str,
    dex3_donor_usd: Path | str,
) -> Path:
    """Compose the certified Dex3 G1 and overlay donor revolute waist roll.

    Ordinary grasp, locomotion, and golden-oracle episodes must keep
    ``create_g1_with_gravity_neutral_dex3``. Only the isolated Psi0 VLA
    interface may load this overlay.
    """
    return _write_composed_g1(
        stable_g1_usd,
        dex3_donor_usd,
        unlock_waist_roll=True,
    )


def _write_composed_g1(
    stable_g1_usd: Path | str,
    dex3_donor_usd: Path | str,
    *,
    unlock_waist_roll: bool,
) -> Path:
    stable_g1 = _require_usd(stable_g1_usd, label="Stable G1")
    dex3_donor = _require_usd(dex3_donor_usd, label="G1 Dex3 donor")
    digest = hashlib.sha256(
        stable_g1.read_bytes() + dex3_donor.read_bytes()
    ).hexdigest()[:16]
    stem = (
        "g1_with_unlocked_waist_roll"
        if unlock_waist_roll
        else "g1_with_gravity_neutral_dex3"
    )
    destination = (
        Path(tempfile.gettempdir()) / "tarsphere" / "robots" / f"{stem}_{digest}.usda"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = _composed_g1_contents(
        stable_g1,
        dex3_donor,
        unlock_waist_roll=unlock_waist_roll,
    )
    if destination.is_file() and destination.read_text(encoding="utf-8") == contents:
        return destination

    temporary = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.tmp.usda")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def require_live_gravity_neutral_g1_dex3(
    stage: object,
    robot_root: str,
) -> None:
    """Fail unless the loaded G1 has physical, gravity-neutral Dex3 hands."""
    from pxr import UsdPhysics

    _require_locomotion_g1_identity(stage, robot_root)
    _require_enabled_body_colliders(stage, robot_root)

    # Waist roll stays fixed like the locomotion-certified asset. Waist pitch
    # is a separate manipulation DOF with its own JointController; it must be
    # revolute here so cuRobo and Isaac can command it outside the 15-D policy.
    roll_joint = stage.GetPrimAtPath(f"{robot_root}/waist_yaw_link/waist_roll_joint")
    if not roll_joint or not roll_joint.IsA(UsdPhysics.FixedJoint):
        raise RuntimeError(
            "The locomotion-certified G1 requires a fixed waist roll joint: "
            f"{robot_root}/waist_yaw_link/waist_roll_joint."
        )
    pitch_joint = stage.GetPrimAtPath(f"{robot_root}/waist_roll_link/waist_pitch_joint")
    if not pitch_joint or not pitch_joint.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(
            "The composed G1 requires a revolute waist pitch joint: "
            f"{robot_root}/waist_roll_link/waist_pitch_joint."
        )

    _require_gravity_neutral_dex3_hands(stage, robot_root)


def require_unlocked_waist_roll(
    stage: object,
    robot_root: str,
) -> None:
    """Fail unless this stage is the isolated VLA roll-unlocked Dex3 G1.

    The certified locomotion composition keeps ``waist_roll_joint`` as a
    FixedJoint. Commanding a 3-DoF VLA waist on that asset is invalid; load
    ``create_g1_with_unlocked_waist_roll`` instead of ignoring roll.
    """
    from pxr import UsdPhysics

    _require_locomotion_g1_identity(stage, robot_root)
    _require_enabled_body_colliders(stage, robot_root)
    roll_path = f"{robot_root}/waist_yaw_link/waist_roll_joint"
    roll_joint = stage.GetPrimAtPath(roll_path)
    if not roll_joint or not roll_joint.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(
            "VLA waist_roll_joint is still FixedJoint or missing. Load "
            "create_g1_with_unlocked_waist_roll; the certified "
            f"locomotion composition keeps roll fixed: {roll_path}."
        )
    pitch_joint = stage.GetPrimAtPath(f"{robot_root}/waist_roll_link/waist_pitch_joint")
    if not pitch_joint or not pitch_joint.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(
            "The composed G1 requires a revolute waist pitch joint: "
            f"{robot_root}/waist_roll_link/waist_pitch_joint."
        )
    _require_gravity_neutral_dex3_hands(stage, robot_root)


def _require_locomotion_g1_identity(stage: object, robot_root: str) -> None:
    imu_path = f"{robot_root}/imu_link"
    if not stage.GetPrimAtPath(imu_path).IsValid():
        raise RuntimeError(
            f"Loaded robot is not the locomotion-certified G1: {imu_path}"
        )


def _require_gravity_neutral_dex3_hands(
    stage: object,
    robot_root: str,
) -> None:
    from pxr import Usd, UsdPhysics

    for hand, link_names in DEX3_LINK_NAMES.items():
        for link_name in link_names:
            link_path = f"{robot_root}/{link_name}"
            link = stage.GetPrimAtPath(link_path)
            if not link or not link.IsValid():
                raise RuntimeError(f"Loaded Dex3 link is missing: {link_path}")
            if not link.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Loaded Dex3 link is not a rigid body: {link_path}")
            mass = UsdPhysics.MassAPI(link).GetMassAttr().Get()
            if mass is None or float(mass) < DEX3_MINIMUM_LINK_MASS_KG:
                raise RuntimeError(
                    "Loaded Dex3 link mass is too small for stable finger "
                    f"dynamics: {link_path} mass={mass!r}, minimum="
                    f"{DEX3_MINIMUM_LINK_MASS_KG:g}."
                )
            diagonal_inertia = UsdPhysics.MassAPI(link).GetDiagonalInertiaAttr().Get()
            if (
                diagonal_inertia is None
                or min(float(value) for value in diagonal_inertia)
                < DEX3_MINIMUM_DIAGONAL_INERTIA_KG_M2
            ):
                raise RuntimeError(
                    "Loaded Dex3 link inertia is too small for stable finger "
                    f"dynamics: {link_path} inertia={diagonal_inertia!r}, "
                    "minimum diagonal component="
                    f"{DEX3_MINIMUM_DIAGONAL_INERTIA_KG_M2:g}."
                )
            gravity_disabled = link.GetAttribute("physxRigidBody:disableGravity").Get()
            if gravity_disabled is not True:
                raise RuntimeError(
                    f"Loaded Dex3 link must be gravity-neutral: {link_path}."
                )
            colliders = [
                prim
                for prim in Usd.PrimRange(
                    link,
                    Usd.TraverseInstanceProxies(),
                )
                if (prim.HasAPI(UsdPhysics.CollisionAPI) and _collision_enabled(prim))
            ]
            if not colliders:
                raise RuntimeError(
                    f"Loaded Dex3 link has no enabled collider: {link_path}"
                )

        for joint_name in DEX3_JOINT_NAMES[hand]:
            joint_path = f"{robot_root}/dex3_joints/{joint_name}"
            joint = stage.GetPrimAtPath(joint_path)
            if not joint or not joint.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Loaded Dex3 joint is missing: {joint_path}")


def _require_enabled_body_colliders(stage: object, robot_root: str) -> None:
    """Fail unless the stable body carries donor collision payloads."""
    from pxr import Usd, UsdPhysics

    missing: list[str] = []
    for link_name in G1_BODY_COLLISION_LINK_NAMES:
        link_path = f"{robot_root}/{link_name}"
        link = stage.GetPrimAtPath(link_path)
        if not link or not link.IsValid():
            missing.append(f"{link_name} (link missing)")
            continue
        body_collisions = stage.GetPrimAtPath(f"{link_path}/body_collisions")
        if not body_collisions or not body_collisions.IsValid():
            missing.append(f"{link_name} (body_collisions missing)")
            continue
        enabled = [
            prim
            for prim in Usd.PrimRange(
                body_collisions,
                Usd.TraverseInstanceProxies(),
            )
            if (prim.HasAPI(UsdPhysics.CollisionAPI) and _collision_enabled(prim))
        ]
        if not enabled:
            missing.append(f"{link_name} (no enabled CollisionAPI)")
    if missing:
        raise RuntimeError(
            "Loaded G1 is missing body collision geometry required to block "
            "furniture penetration. The locomotion-certified USD only ships "
            "ankle Mesh colliders; donor body payloads must be remounted. "
            f"Missing: {missing!r}."
        )


def _collision_enabled(prim: object) -> bool:
    """Treat an unauthored collisionEnabled opinion as the USD default (on)."""
    from pxr import UsdPhysics

    value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    return True if value is None else bool(value)


def _require_usd(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} USD is missing: {resolved}")
    return resolved


def _usda_path(path: Path) -> str:
    text = path.as_posix()
    if "@" in text or "\n" in text or "\r" in text:
        raise ValueError(f"USD path cannot be represented in USDA: {path}")
    return text


def _usda_color3f(rgb: tuple[float, float, float]) -> str:
    return "({})".format(", ".join(f"{channel:.2f}" for channel in rgb))


def _usda_scalar(value: float) -> str:
    return f"{value:.2f}"


def _g1_visual_looks_overlay() -> str:
    """USDA Looks overrides and the composed Dex3 material. Visual only."""
    dark = _usda_color3f(G1_LOOK_DARK_DIFFUSE)
    white = _usda_color3f(G1_LOOK_WHITE_DIFFUSE)
    dex3 = _usda_color3f(G1_LOOK_DEX3_DIFFUSE)
    return f'''    over "Looks"
    {{
        over "dark"
        {{
            over "Shader"
            {{
                color3f inputs:diffuse_color_constant = {dark}
                float inputs:reflection_roughness_constant = {_usda_scalar(G1_LOOK_DARK_ROUGHNESS)}
                float inputs:metallic_constant = {_usda_scalar(G1_LOOK_DARK_METALLIC)}
            }}
        }}

        over "white"
        {{
            over "Shader"
            {{
                color3f inputs:diffuse_color_constant = {white}
                float inputs:reflection_roughness_constant = {_usda_scalar(G1_LOOK_WHITE_ROUGHNESS)}
                float inputs:metallic_constant = {_usda_scalar(G1_LOOK_WHITE_METALLIC)}
            }}
        }}

        over "material_DefaultMaterial_1"
        {{
            over "Shader"
            {{
                color3f inputs:diffuse_color_constant = {dark}
                float inputs:reflection_roughness_constant = {_usda_scalar(G1_LOOK_DARK_ROUGHNESS)}
                float inputs:metallic_constant = {_usda_scalar(G1_LOOK_DARK_METALLIC)}
            }}
        }}

        def Material "{G1_LOOK_DEX3_NAME}"
        {{
            token outputs:mdl:displacement.connect = <{G1_LOOK_DEX3_PRIM}/Shader.outputs:out>
            token outputs:mdl:surface.connect = <{G1_LOOK_DEX3_PRIM}/Shader.outputs:out>
            token outputs:mdl:volume.connect = <{G1_LOOK_DEX3_PRIM}/Shader.outputs:out>

            def Shader "Shader"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = {dex3}
                float inputs:reflection_roughness_constant = {_usda_scalar(G1_LOOK_DEX3_ROUGHNESS)}
                float inputs:metallic_constant = {_usda_scalar(G1_LOOK_DEX3_METALLIC)}
                token outputs:out
            }}
        }}
    }}'''


def _finger_joint_bodies(
    hand: str,
    joint_name: str,
) -> tuple[str, str]:
    stem = joint_name.removeprefix(f"{hand}_hand_").removesuffix("_joint")
    finger, segment_text = stem.rsplit("_", maxsplit=1)
    segment = int(segment_text)
    child = f"{hand}_hand_{finger}_{segment}_link"
    if segment == 0:
        parent = f"{hand}_hand_palm_link"
    else:
        parent = f"{hand}_hand_{finger}_{segment - 1}_link"
    return parent, child


def _composed_g1_contents(
    stable_g1: Path,
    dex3_donor: Path,
    *,
    unlock_waist_roll: bool = False,
) -> str:
    stable_text = _usda_path(stable_g1)
    donor_text = _usda_path(dex3_donor)
    link_definitions = []
    for link_names in DEX3_LINK_NAMES.values():
        for link_name in link_names:
            link_definitions.append(
                f'''    def Xform "{link_name}" (
        prepend apiSchemas = ["PhysxRigidBodyAPI", "MaterialBindingAPI"]
        prepend references = @{donor_text}@<{G1_DEX3_DONOR_ROOT_PRIM}/{link_name}>
    )
    {{
        bool physxRigidBody:disableGravity = 1
        rel material:binding:physics = <{G1_STABLE_ROOT_PRIM}/dex3_grasp_material> (
            bindMaterialAs = "strongerThanDescendants"
        )
        rel material:binding = <{G1_LOOK_DEX3_PRIM}> (
            bindMaterialAs = "strongerThanDescendants"
        )
    }}'''
            )

    body_collision_definitions = []
    for link_name in G1_BODY_COLLISION_LINK_NAMES:
        disable_loco_foot = ""
        if link_name in G1_LOCO_FOOT_COLLISION_LINKS:
            # Loco USD authors a Mesh prim named ``collisions``. Keep the prim
            # for layer compatibility but turn it off once donor ankle hulls
            # are present under ``body_collisions``.
            disable_loco_foot = """
        over "collisions"
        {
            bool physics:collisionEnabled = 0
        }
"""
        body_collision_definitions.append(
            f'''    over "{link_name}"
    {{{disable_loco_foot}
        def Xform "body_collisions" (
            prepend references = @{donor_text}@<{G1_DEX3_DONOR_ROOT_PRIM}/{link_name}/collisions>
        )
        {{
        }}
    }}'''
        )

    joint_definitions = []
    for hand in ("left", "right"):
        donor_palm_joint = f"{G1_DEX3_DONOR_ROOT_PRIM}/joints/{hand}_hand_palm_joint"
        joint_definitions.append(
            f'''        def PhysicsFixedJoint "{hand}_dex3_palm_joint" (
            prepend references = @{donor_text}@<{donor_palm_joint}>
        )
        {{
            rel physics:body0 = <{G1_STABLE_ROOT_PRIM}/{hand}_wrist_yaw_link>
            rel physics:body1 = <{G1_STABLE_ROOT_PRIM}/{hand}_hand_palm_link>
        }}'''
        )
        for joint_name in DEX3_JOINT_NAMES[hand]:
            parent, child = _finger_joint_bodies(hand, joint_name)
            donor_joint = f"{G1_DEX3_DONOR_ROOT_PRIM}/joints/{joint_name}"
            joint_definitions.append(
                f'''        def PhysicsRevoluteJoint "{joint_name}" (
            prepend references = @{donor_text}@<{donor_joint}>
        )
        {{
            rel physics:body0 = <{G1_STABLE_ROOT_PRIM}/{parent}>
            rel physics:body1 = <{G1_STABLE_ROOT_PRIM}/{child}>
        }}'''
            )

    links = "\n\n".join(link_definitions)
    body_collisions = "\n\n".join(body_collision_definitions)
    joints = "\n\n".join(joint_definitions)
    donor_pitch_joint = f"{G1_DEX3_DONOR_ROOT_PRIM}/joints/waist_pitch_joint"
    waist_pitch_joint = f"""    over "waist_roll_link"
    {{
        def PhysicsRevoluteJoint "waist_pitch_joint" (
            prepend references = @{donor_text}@<{donor_pitch_joint}>
        )
        {{
            rel physics:body0 = <{G1_STABLE_ROOT_PRIM}/waist_roll_link>
            rel physics:body1 = <{G1_STABLE_ROOT_PRIM}/torso_link>
            point3f physics:localPos0 = {G1_STABLE_WAIST_PITCH_LOCAL_POS0}
            point3f physics:localPos1 = (0, 0, 0)
            float drive:angular:physics:stiffness = {G1_WAIST_PITCH_DRIVE_STIFFNESS}
            float drive:angular:physics:damping = {G1_WAIST_PITCH_DRIVE_DAMPING}
            float drive:angular:physics:maxForce = {G1_WAIST_PITCH_DRIVE_MAX_FORCE}
        }}
    }}"""
    # Roll and pitch are manipulation DOFs. The certified composition holds
    # roll as FixedJoint at 0. The VLA interface overlays the donor revolute here and
    # commands it only on that path; ordinary loco never sees this block.
    waist_roll_overlay = ""
    if unlock_waist_roll:
        donor_roll_joint = f"{G1_DEX3_DONOR_ROOT_PRIM}/joints/waist_roll_joint"
        waist_roll_overlay = f"""    over "waist_yaw_link"
    {{
        def PhysicsRevoluteJoint "waist_roll_joint" (
            prepend references = @{donor_text}@<{donor_roll_joint}>
        )
        {{
            rel physics:body0 = <{G1_STABLE_ROOT_PRIM}/waist_yaw_link>
            rel physics:body1 = <{G1_STABLE_ROOT_PRIM}/waist_roll_link>
            point3f physics:localPos0 = {G1_STABLE_WAIST_ROLL_LOCAL_POS0}
            point3f physics:localPos1 = (0, 0, 0)
            float drive:angular:physics:stiffness = {G1_WAIST_ROLL_DRIVE_STIFFNESS}
            float drive:angular:physics:damping = {G1_WAIST_ROLL_DRIVE_DAMPING}
            float drive:angular:physics:maxForce = {G1_WAIST_ROLL_DRIVE_MAX_FORCE}
        }}
    }}

"""
    return f"""#usda 1.0
(
    defaultPrim = "g1_29dof"
)

def Xform "g1_29dof" (
    prepend references = @{stable_text}@<{G1_STABLE_ROOT_PRIM}>
)
{{
    over "pelvis" (
        prepend apiSchemas = ["PhysxContactReportAPI"]
    )
    {{
        float physxContactReport:threshold = 0
    }}

    over "left_rubber_hand"
    {{
        token visibility = "invisible"
    }}

    over "right_rubber_hand"
    {{
        token visibility = "invisible"
    }}

{_g1_visual_looks_overlay()}

    def Material "dex3_grasp_material" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {DEX3_STATIC_FRICTION}
        float physics:dynamicFriction = {DEX3_DYNAMIC_FRICTION}
        float physics:restitution = 0
    }}

{body_collisions}

{waist_roll_overlay}{waist_pitch_joint}

{links}

    def Scope "dex3_joints"
    {{
{joints}
    }}
}}
"""


__all__ = [
    "G1_BODY_COLLISION_LINK_NAMES",
    "G1_DEX3_DONOR_ROOT_PRIM",
    "G1_LOOK_DARK_DIFFUSE",
    "G1_LOOK_DARK_METALLIC",
    "G1_LOOK_DARK_ROUGHNESS",
    "G1_LOOK_DEX3_DIFFUSE",
    "G1_LOOK_DEX3_METALLIC",
    "G1_LOOK_DEX3_NAME",
    "G1_LOOK_DEX3_PRIM",
    "G1_LOOK_DEX3_ROUGHNESS",
    "G1_LOOK_WHITE_DIFFUSE",
    "G1_LOOK_WHITE_METALLIC",
    "G1_LOOK_WHITE_ROUGHNESS",
    "G1_LOCO_FOOT_COLLISION_LINKS",
    "G1_STABLE_ROOT_PRIM",
    "create_g1_with_gravity_neutral_dex3",
    "create_g1_with_unlocked_waist_roll",
    "require_unlocked_waist_roll",
    "require_live_gravity_neutral_g1_dex3",
]
