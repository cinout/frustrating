import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET
import numpy as np

# from fsdet.utils.file_io import PathManager

from iopath.common.file_io import (
    HTTPURLHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManager as PathManagerBase,
)

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class FsDetHandler(PathHandler):
    """
    Resolve anything that's in FsDet model zoo.
    """

    PREFIX = "fsdet://"
    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.URL_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())
PathManager.register_handler(FsDetHandler())


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "nectarine",
    "orange",
    "cereal",
    "almond_mix",
]  # FIXME: create new classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in VOC_CLASSES}

    with PathManager.open("datasets/mvtec/ImageSets/Main/trainval.txt") as f:
        fileids = np.loadtxt(f, dtype=np.str).tolist()
        data.extend(fileids)

    for fileid in data:
        # year = "2012" if "_" in fileid else "2007"
        # dirname = os.path.join("datasets", "VOC{}".format(year))

        anno_file = os.path.join(
            "datasets/mvtec", "Annotations", fileid + ".xml"
        )
        tree = ET.parse(anno_file)
        clses = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            clses.append(
                cls
            )  # TODO: find all classes of object in this anno_file
        for cls in set(clses):
            data_per_cat[cls].append(
                anno_file
            )  # all annotation file paths by classname

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5]  # FIXME: change later [1, 2, 3, 5, 10]

    # we don't care seeds here, refer to prepare_pascol_xxx.py if you want to use different seeds
    random.seed(0)
    for c in data_per_cat.keys():  # TODO: for each classname
        c_data = []
        for j, shot in enumerate(shots):  # TODO: for each shot number
            diff_shot = shots[j] - shots[j - 1] if j != 0 else 1
            shots_c = random.sample(
                data_per_cat[c], diff_shot
            )  # TODO: anno file paths for additional shots
            num_objs = 0
            for s in shots_c:
                print("=================")
                print(s, c_data)
                if s not in c_data:
                    tree = ET.parse(s)
                    file = tree.find("filename").text  # contains suffix
                    name = "datasets/mvtec/JPEGImages/{}".format(
                        file
                    )  # TODO: image file path
                    c_data.append(name)
                    for obj in tree.findall("object"):
                        if (
                            obj.find("name").text == c
                        ):  # count objects for current class
                            num_objs += 1
                    if num_objs >= diff_shot:
                        break
            result[c][shot] = copy.deepcopy(
                c_data
            )  # TODO: image file paths by (1) classname (2) #shot

    save_path = "datasets/mvtecsplit"
    os.makedirs(save_path, exist_ok=True)
    for c in result.keys():
        for shot in result[c].keys():
            filename = "box_{}shot_{}_train.txt".format(shot, c)
            with open(os.path.join(save_path, filename), "w") as fp:
                fp.write("\n".join(result[c][shot]) + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)
