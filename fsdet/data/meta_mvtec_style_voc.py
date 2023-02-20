import os, json
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager

__all__ = ["register_meta_mvtec_style_voc"]


def load_mvtec_instances(name: str, dirname: str, split: str, classnames: str):
    """
    to Detectron2 format.
    Args:
        name: e.g., mvtec_trainval_novel_1shot
        dirname: e.g., datasets/mvtec
        split (str): one of "train", "test", "val", "trainval", only used in base training
        classnames: a list of class names (all/base/novel), defined in builtin_meta.py
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join(
            "datasets", "mvtecsplit"
        )  # TODO: the file path stucture is created in file prepare_mvtec_few_shot_style_voc.py
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(
                        ".jpg" if fid.endswith("jpg") else ".png"
                    )[0]
                    for fid in fileids_
                ]  # TODO: list of image file ids without .jpg suffix
                fileids[
                    cls
                ] = fileids_  # TODO: dictionary, with "key" of classname
    else:
        if name != "mvtec_test_all":  # FIXME: update
            with PathManager.open(
                os.path.join(dirname, "ImageSets", "Main", split + ".txt")
            ) as f:
                fileids = np.loadtxt(f, dtype=np.str)  # TODO: image file ids

    dicts = []
    if is_shots:
        # TODO: few shot fine tuning
        for cls, fileids_ in fileids.items():
            dicts_ = []  # dicts_ for each class
            for fileid in fileids_:
                dirname = os.path.join("datasets", "mvtec")
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname,
                    "JPEGImages",
                    fileid
                    + (".png" if fileid.startswith("mvtec") else ".jpg"),
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        # TODO: the object does not belong to the cls
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0  # TODO: what is this -1.0 for?
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(
                    dicts_, int(shot), replace=False
                )  # TODO: "replace=False" means one value can be selected only once
            dicts.extend(dicts_)
    else:
        # TODO: base training
        if name == "mvtec_test_all":  # FIXME: update
            test_samples_path = os.path.join(dirname, "mvtec_novel_samples")
            for file in os.listdir(test_samples_path):
                file_name = os.path.join(test_samples_path, file)
                image_id = file.split(".png")[0]
                r = {
                    "file_name": file_name,
                    "image_id": image_id,
                    "height": int(
                        1280
                    ),  # FIXME: update: breakfast:1280, screw:1100
                    "width": int(1600),  # FIXME: update: breakfast/screw:1600
                    "annotations": [],
                }
                dicts.append(r)
        else:
            for fileid in fileids:
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname,
                    "JPEGImages",
                    fileid
                    + (".png" if fileid.startswith("mvtec") else ".jpg"),
                )

                tree = ET.parse(anno_file)

                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }
                instances = []

                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    if not (cls in classnames):
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances.append(
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    )
                r["annotations"] = instances
                dicts.append(r)
    return dicts


def register_meta_mvtec_style_voc(name, metadata, dirname, split, keepclasses):
    # TODO: split: only used in base training (train/test/val/trainval)
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"]

    # TODO: register dataset (step 1)
    DatasetCatalog.register(
        name,  # TODO: name of dataset, this will be used in the config file
        lambda: load_mvtec_instances(
            name, dirname, split, thing_classes
        ),  # TODO: call your dataset loader to get the data
    )

    # TODO: register meta information (step 2)
    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        split=split,
        year=2007,
        base_classes=metadata["base_classes"],
        novel_classes=metadata["novel_classes"],
    )
