from .btmri import BTMRI
from .busi import BUSI
from .ctkidney import CTKidney
from .kneexray import KneeXray
from .kvasir import Kvasir
from .lungcolon import LungColon
from .retina import RETINA
from .covid import COVID_19
from .dermamnist import DermaMNIST
from .octmnist import OCTMNIST
from .chmnist import CHMNIST
from .btmri_p import BTMRI_P
from .btmri_s import BTMRI_S
from .brisc import BRISC
from .buid import BUID
from .busbra import BUSBRA
from .udiat import UDIAT

dataset_list = {
                "BUSI": BUSI,
                "BUID": BUID,
                "BUSBRA": BUSBRA,
                "UDIAT": UDIAT,
                "BTMRI": BTMRI,
                "BTMRI_P": BTMRI_P,
                "BTMRI_S": BTMRI_S,
                "BRISC": BRISC,
                "CTKidney": CTKidney,
                "KneeXray": KneeXray,
                "Kvasir": Kvasir,
                "LungColon": LungColon,
                "RETINA": RETINA,
                "COVID_19": COVID_19,
                "DermaMNIST": DermaMNIST,
                "OCTMNIST": OCTMNIST,
                "CHMNIST": CHMNIST
                }


def build_dataset(cfg):
    return dataset_list[cfg.DATASET.NAME](cfg)