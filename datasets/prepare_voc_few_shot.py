import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from fsdet.utils.file_io import PathManager

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 20], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data = []
    data_per_cat = {c: [] for c in VOC_CLASSES}
    for year in [2007, 2012]:
        data_file = "datasets/VOC{}/ImageSets/Main/trainval.txt".format(year)
        with PathManager.open(data_file) as f:
            fileids = np.loadtxt(f, dtype=np.str).tolist()
        data.extend(fileids)
    for fileid in data:
        year = "2012" if "_" in fileid else "2007"
        dirname = os.path.join("datasets", "VOC{}".format(year))
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
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
            )  # TODO: all annotation file paths by classname

    result = {cls: {} for cls in data_per_cat.keys()}
    shots = [1, 2, 3, 5, 10]
    for i in range(args.seeds[0], args.seeds[1]):  # TODO: for each seed
        random.seed(i)
        for c in data_per_cat.keys():  # TODO: for each classname
            c_data = []
            for j, shot in enumerate(shots):  # TODO: for each shot number
                diff_shot = shots[j] - shots[j - 1] if j != 0 else 1
                shots_c = random.sample(
                    data_per_cat[c], diff_shot
                )  # TODO: anno file paths for additional shots #FIXME: how does it make sure it won't sample duplicate file from last loop?
                num_objs = 0
                for s in shots_c:
                    print(f"s: {s}, c_data: {c_data}")
                    if (
                        s not in c_data
                    ):  # FIXME: does this comparison make sense?? c_data contains image file path, while s is annotation file path
                        tree = ET.parse(s)
                        file = tree.find("filename").text  # contains suffix
                        year = tree.find("folder").text
                        name = "datasets/{}/JPEGImages/{}".format(
                            year, file
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
        save_path = "datasets/vocsplit/seed{}".format(i)
        os.makedirs(save_path, exist_ok=True)
        for c in result.keys():
            for shot in result[c].keys():
                filename = "box_{}shot_{}_train.txt".format(shot, c)
                with open(os.path.join(save_path, filename), "w") as fp:
                    fp.write("\n".join(result[c][shot]) + "\n")


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)
