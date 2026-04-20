import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Base path for Google Scanned Objects (USD format, already converted)
_OBJ_BASE = "/home/leesu37/AP-project/object_usd/google_objects_usd"

# 20 objects randomly selected from Google Scanned Objects (seed=42).
# Each directory contains:
#   <name>.usd              — main USD asset
#   instanceable_meshes.usd — instancing용 메쉬
#   materials/              — 텍스처 파일
_OBJECT_USDS = [ # object 20 fixed
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
]  # 총 20개 (Google Scanned Objects, seed=42)

# pile_XX → 오브젝트 이름 (USD 파일명에서 확장자 제거)
# ex) pile_15 → "Hey_You_Pikachu_Nintendo_64"
PILE_TO_OBJECT_NAME: dict[str, str] = {
    f"pile_{i:02d}": _OBJECT_USDS[i].split("/")[-1].replace(".usd", "")
    for i in range(len(_OBJECT_USDS))
}


# R1 Pro 초기 관절 자세는 r1pro.py의 R1PRO_HIGH_PD_CFG.init_state에 정의됨


def _make_pile_cfg(idx: int) -> RigidObjectCfg:
    """Create a RigidObjectCfg for pile_XX using a Lightwheel OpenSource USD.

    Lightwheel USDs are non-instanceable → collision_props, mass_props를
    Isaac Lab API로 직접 override 가능.
    """
    usd_path = _OBJECT_USDS[idx]
    obj_name = usd_path.split("/")[-2]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/pile_{idx:02d}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, # 높일수록 penetration 감소 (4→16)
                solver_velocity_iteration_count=4,  # 속도 계산 반복 횟수 (1→4)
                max_angular_velocity=100.0,         # 회전속도 상한
                max_linear_velocity=100.0,          # 선속도 상한
                max_depenetration_velocity=5.0,     # 겹침 해소 속도 높임 (1→5)
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # GSO 일반 물체 평균 100g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,  # 충돌 감지 시작 거리 (작을수록 정밀)
                rest_offset=0.0,       # 접촉 시 유지할 최소 간격
            ),
            semantic_tags=[("class", obj_name)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)), # 중력으로 낙하 → pile 형성
    )


@configclass
class OcclusionSceneCfg(InteractiveSceneCfg):

    # 바닥 (테이블 높이 1.05m를 고려해 z=-1.05에 배치)
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # 테이블 (scale=1.5 로 가로/세로 1.5배 확대)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.5, 1.0),  # XY 1.5배 확대, 높이는 유지
        ),
    )

    # Franka Panda — 테이블 표면(z=0) 투명벽 밖(-x)에 배치, pile(x=0.3~0.7) 방향(+x)으로 향함
    # wall_x_neg: x=-0.1, 로봇 베이스 x=-0.2 → 벽 밖 10cm, pile까지 거리 ~0.5~0.9m → 닿음
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

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=800.0),
    )

    # 경계 벽: 1.2×1.2m 정사각형 (중심=(0.5, 0), 높이 12cm, 두께 2cm), 투명 유리 재질
    # X: 0.5 ± 0.6 → [-0.1, 1.1],  Y: 0 ± 0.6 → [-0.6, 0.6]
    wall_x_neg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallXNeg",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0.0, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 1.20, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_x_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallXPos",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1, 0.0, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 1.20, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_y_neg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallYNeg",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.6, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.02, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )
    wall_y_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallYPos",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.6, 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.02, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.2),
            visible=True,
        ),
    )

    # Lightwheel OpenSource Manipulation 20개 (pile_00 ~ pile_19)
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

    # overhead 카메라 (pile 수직 내려보기)
    # 위치: pile 중심(x=0.5) 바로 위 2.5m → SDT deproject 정확
    # focal_length=16 → 수평 FOV ~66°, 2.5m 높이에서 가로 약 3.2m 커버
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 2.5),   # reset()에서 set_world_poses_from_view로 덮어씌움
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # wrist 카메라 — panda_hand에 부착, 그리퍼 전방을 바라봄 (GR00T wrist_image_left)
    # pos: hand 로컬 -Z 방향으로 0.08m (손목 쪽으로 후퇴) → 그리퍼 앞 공간 커버
    # rot: ROS 컨벤션, 광축(+Z)을 hand +Z(gripper approach)에 정렬 → identity
    # focal_length=5 → FOV ~90°, 근거리(~0.5m) 작업공간 커버
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_period=0.1,
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
class OcclusionEnvCfg:
    # 씬
    scene: OcclusionSceneCfg = OcclusionSceneCfg(num_envs=4, env_spacing=2.0) #돌아가는 환경 개수 4개

    # 시뮬레이션
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,  # dt 절반으로 줄임 → 고속 충돌 시 penetration 감소
        physx=sim_utils.PhysxCfg(
            solver_type=1,                  # 1=TGS (물체 쌓임에 더 안정적), 0=PGS
            enable_ccd=True,                # 고속 이동 물체의 tunneling 방지
            bounce_threshold_velocity=0.2,  # 이 속도 이하면 bounce 무시 → 안정적 pile
        ),
    )

    # 에피소드 길이
    episode_length_s: float = 5.0
