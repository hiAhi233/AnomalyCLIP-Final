"""Basic concept descriptions for BUSI anomaly detection."""

from collections import OrderedDict

BUSI_NORMAL_CONCEPTS = OrderedDict({
    "normal_tissue": [
        "homogeneous echotexture with uniform echogenicity throughout the breast tissue",
        "乳腺组织回声均匀，质地一致，未见局灶性回声改变",
        "uniformly textured breast parenchyma showing consistent echo pattern",
    ],
    "regular_architecture": [
        "regular tissue architecture with well-organized tissue planes",
        "组织结构规则，层次清晰，组织界面分明",
    ],
    "isoechoic_texture": [
        "isoechoic texture with echogenicity comparable to surrounding tissue",
        "等回声质地，回声强度与周围参照组织一致",
    ],
})

BUSI_ANOMALY_CONCEPTS = OrderedDict({
    "inflammation": [
        "localized inflammatory change with edema and increased vascularity",
        "局灶性炎性改变，伴组织水肿和血供增加",
    ],
    "tumor_mass": [
        "a space-occupying solid lesion with mass effect on surrounding structures",
        "占位性实性病变，对周围结构产生推挤效应",
    ],
    "necrosis": [
        "area of tissue necrosis with loss of normal cellular architecture",
        "组织坏死区域，正常细胞结构丧失",
    ],
    "calcification": [
        "dense calcific deposit appearing as hyperechoic focus",
        "致密钙化沉积，呈强回声灶",
    ],
    "cyst_fluid": [
        "well-circumscribed fluid-filled cystic structure with thin walls",
        "边界清晰的含液囊性结构，壁薄",
    ],
    "fibrosis_scar": [
        "fibrotic tissue with linear patterns replacing normal parenchyma",
        "纤维化组织呈线状替代正常实质",
    ],
})

TEXT_DESCRIPTIONS_REGISTRY = {
    "BUSI": {"normal": BUSI_NORMAL_CONCEPTS, "anomaly": BUSI_ANOMALY_CONCEPTS},
}


def get_text_descriptions(dataset_name):
    if dataset_name not in TEXT_DESCRIPTIONS_REGISTRY:
        # 未知数据集返回 BUSI 的通用概念描述作为回退
        return TEXT_DESCRIPTIONS_REGISTRY["BUSI"]
    return TEXT_DESCRIPTIONS_REGISTRY[dataset_name]
