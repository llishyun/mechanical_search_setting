import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

_OBJ_BASE = (
    "/home/leesu37/AP-project/mechanical_search_setting"
    "/mechanical_search_isaac_sh/assets/object_usd/google_objects_usd"
)

_OBJECT_USDS = [
    f"{_OBJ_BASE}/Cole_Hardware_Saucer_Electric/Cole_Hardware_Saucer_Electric.usd",
    f"{_OBJ_BASE}/Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R/Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R.usd",
    f"{_OBJ_BASE}/LADYBUG_BEAD/LADYBUG_BEAD.usd",
    f"{_OBJ_BASE}/Hyaluronic_Acid/Hyaluronic_Acid.usd",
    f"{_OBJ_BASE}/Grreat_Choice_Dog_Double_Dish_Plastic_Blue/Grreat_Choice_Dog_Double_Dish_Plastic_Blue.usd",
    f"{_OBJ_BASE}/Design_Ideas_Drawer_Store_Organizer/Design_Ideas_Drawer_Store_Organizer.usd",
    f"{_OBJ_BASE}/Cole_Hardware_Electric_Pot_Assortment_55/Cole_Hardware_Electric_Pot_Assortment_55.usd",
    f"{_OBJ_BASE}/Threshold_Basket_Natural_Finish_Fabric_Liner_Small/Threshold_Basket_Natural_Finish_Fabric_Liner_Small.usd",
    f"{_OBJ_BASE}/Cole_Hardware_Bowl_Scirocco_YellowBlue/Cole_Hardware_Bowl_Scirocco_YellowBlue.usd",
    f"{_OBJ_BASE}/UGG_Bailey_Bow_Womens_Clogs_Black_7/UGG_Bailey_Bow_Womens_Clogs_Black_7.usd",
    f"{_OBJ_BASE}/Razer_Abyssus_Ambidextrous_Gaming_Mouse/Razer_Abyssus_Ambidextrous_Gaming_Mouse.usd",
    f"{_OBJ_BASE}/Balderdash_Game/Balderdash_Game.usd",
    f"{_OBJ_BASE}/BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup/BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup.usd",
    f"{_OBJ_BASE}/Cole_Hardware_Dishtowel_Blue/Cole_Hardware_Dishtowel_Blue.usd",
    f"{_OBJ_BASE}/Granimals_20_Wooden_ABC_Blocks_Wagon/Granimals_20_Wooden_ABC_Blocks_Wagon.usd",
    f"{_OBJ_BASE}/Hey_You_Pikachu_Nintendo_64/Hey_You_Pikachu_Nintendo_64.usd",
    f"{_OBJ_BASE}/Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White/Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White.usd",
    f"{_OBJ_BASE}/W_Lou_z0dkC78niiZ/W_Lou_z0dkC78niiZ.usd",
    f"{_OBJ_BASE}/Avengers_Thor_PLlrpYniaeB/Avengers_Thor_PLlrpYniaeB.usd",
    f"{_OBJ_BASE}/Threshold_Porcelain_Serving_Bowl_Coupe_White/Threshold_Porcelain_Serving_Bowl_Coupe_White.usd",
]

NUM_OBJECTS = len(_OBJECT_USDS)

# Scene-level name for each pile object (key into InteractiveScene)
PILE_NAMES: list[str] = [f"pile_{i:02d}" for i in range(NUM_OBJECTS)]

# pile_XX → USD directory name (used as object identifier / semantic class)
PILE_TO_OBJECT_NAME: dict[str, str] = {
    f"pile_{i:02d}": _OBJECT_USDS[i].split("/")[-2]
    for i in range(NUM_OBJECTS)
}

# Drop zone above table surface (z=0).
# x: pile center 0.5 ± 0.20 m
# y: pile center 0.0 ± 0.20 m
# z: 5–15 cm above table (벽 높이 12cm 이하로 튀지 않도록)
DROP_X = (0.30, 0.70)
DROP_Y = (-0.20, 0.20)
DROP_Z = (0.05, 0.15)

# Physics settle steps after object drop
SETTLE_STEPS = 600 #물리 시뮬레이션이 안정화될때까지 기다리는 스텝후, 600step 동안은 로봇은 대기


def _make_pile_cfg(idx: int) -> RigidObjectCfg:
    usd_path = _OBJECT_USDS[idx]
    obj_name = usd_path.split("/")[-2]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/pile_{idx:02d}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            semantic_tags=[("class", obj_name)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.8)),
    )


@configclass
class MechSearchSceneCfg(InteractiveSceneCfg):

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.5, 1.0),
        ),
    )

    # Franka base at x=-0.2 (outside wall_x_neg at x=-0.1), faces +x toward pile
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.2, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=800.0),
    )

    # Boundary walls: 1.0×1.0 m square, centre=(0.5, 0), height=12 cm, semi-transparent
    wall_x_neg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallXNeg",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 1.00, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_x_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallXPos",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 1.00, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_y_neg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallYNeg",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.5, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(1.00, 0.02, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_y_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallYPos",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.5, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(1.00, 0.02, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )

    pile_00: RigidObjectCfg = _make_pile_cfg(0)
    pile_01: RigidObjectCfg = _make_pile_cfg(1)
    pile_02: RigidObjectCfg = _make_pile_cfg(2)
    pile_03: RigidObjectCfg = _make_pile_cfg(3)
    pile_04: RigidObjectCfg = _make_pile_cfg(4)
    pile_05: RigidObjectCfg = _make_pile_cfg(5)
    pile_06: RigidObjectCfg = _make_pile_cfg(6)
    pile_07: RigidObjectCfg = _make_pile_cfg(7)
    pile_08: RigidObjectCfg = _make_pile_cfg(8)
    pile_09: RigidObjectCfg = _make_pile_cfg(9)
    pile_10: RigidObjectCfg = _make_pile_cfg(10)
    pile_11: RigidObjectCfg = _make_pile_cfg(11)
    pile_12: RigidObjectCfg = _make_pile_cfg(12)
    pile_13: RigidObjectCfg = _make_pile_cfg(13)
    pile_14: RigidObjectCfg = _make_pile_cfg(14)
    pile_15: RigidObjectCfg = _make_pile_cfg(15)
    pile_16: RigidObjectCfg = _make_pile_cfg(16)
    pile_17: RigidObjectCfg = _make_pile_cfg(17)
    pile_18: RigidObjectCfg = _make_pile_cfg(18)
    pile_19: RigidObjectCfg = _make_pile_cfg(19)

    # Overhead camera — looks straight down at pile centre from 2.5 m
    # update_period=0: manual update only (prevents stale-frame timing bugs)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 2.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # Wrist camera — attached to panda_hand, optical axis along gripper approach (+Z)
    # update_period=0: manual update only
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_period=0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=0.5,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.08),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


@configclass
class MechSearchEnvCfg:
    scene: MechSearchSceneCfg = MechSearchSceneCfg(num_envs=1, env_spacing=3.0)

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_ccd=True,
            bounce_threshold_velocity=0.5,
        ),
    )

    episode_length_s: float = 10.0
