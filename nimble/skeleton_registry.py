from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    description: str
    loader: Callable[[], Any]
    ik_mapping: Tuple[Tuple[str, int], ...] = ()
    foot_body_names: Tuple[str, str] = ('calcn_l', 'calcn_r')
    unmapped_proxies: Tuple[Tuple[int, str, str], ...] = ()
    notes: str = ''

def _load_rajagopal() -> Any:
    from nimblephysics.models import rajagopal as raj
    return raj.RajagopalHumanBodyModel()
_RAJAGOPAL_IK: Tuple[Tuple[str, int], ...] = (('ground_pelvis', 0), ('hip_l', 1), ('walker_knee_l', 4), ('ankle_l', 7), ('mtp_l', 10), ('hip_r', 2), ('walker_knee_r', 5), ('ankle_r', 8), ('mtp_r', 11), ('back', 3), ('acromial_l', 16), ('elbow_l', 18), ('radius_hand_l', 20), ('acromial_r', 17), ('elbow_r', 19), ('radius_hand_r', 21))
_RAJAGOPAL_PROXIES: Tuple[Tuple[int, str, str], ...] = ((6, 'joint', 'back'), (9, 'joint', 'back'), (12, 'joint', 'back'), (13, 'joint', 'acromial_l'), (14, 'joint', 'acromial_r'), (15, 'body', 'torso'))
SKELETONS: Dict[str, SkeletonSpec] = {'rajagopal': SkeletonSpec(name='rajagopal', description='Bundled Rajagopal 2015. 37 DOF; 16 HML joints mapped.', loader=_load_rajagopal, ik_mapping=_RAJAGOPAL_IK, unmapped_proxies=_RAJAGOPAL_PROXIES, notes='Built into nimblephysics. No download required.')}

def available_skeletons() -> List[str]:
    return sorted(SKELETONS.keys())

def get_spec(name: str='rajagopal') -> SkeletonSpec:
    key = str(name).strip().lower()
    if key not in SKELETONS:
        raise KeyError(f'Unknown skeleton {name!r}; available: {available_skeletons()}')
    return SKELETONS[key]

def load_skeleton(name: str='rajagopal', *, with_geometry: bool=False) -> Tuple[Any, SkeletonSpec]:
    del with_geometry
    spec = get_spec(name)
    return (spec.loader(), spec)

def list_joint_names(skeleton: Any) -> List[str]:
    return [skeleton.getJoint(i).getName() for i in range(skeleton.getNumJoints())]

def list_body_names(skeleton: Any) -> List[str]:
    return [skeleton.getBodyNode(i).getName() for i in range(skeleton.getNumBodyNodes())]