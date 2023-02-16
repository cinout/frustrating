import os, json
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.utils.file_io import PathManager

__all__ = ["register_meta_mvtec_style_voc"]

# FIXME: update more shots and more classes
mvtec_matadata = {
    "1shot": {
        "mvtec_breakfast_box": {
            "images": ["mvtec_breakfast_box_train_006.png"],
            "annotation": "annotations_mvtec_breakfast_box.json",
        }
    }
}


def load_mvtec_instances(name: str, dirname: str, split: str, classnames: str):
    """
    to Detectron2 format.
    Args:
        name: e.g., "voc_2007_trainval_base1" or "voc_2007_trainval_novel1_3shot_seed10"
        dirname: e.g., VOC2007
        split (str): one of "train", "test", "val", "trainval"
        classnames: a list of class names (all/base/novel) based on split1/2/3, defined in builtin_meta.py
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join(
            "datasets", "vocsplit"
        )  # TODO: refer to http://dl.yf.io/fs-det/, the file path stucture is created in file prepare_mvtec_few_shot_style_voc.py
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            # FIXME: get files from self-created MVTEC dataset
            if cls in [
                "nectarine",
                "orange",
                "cereal",
                "almond_mix",
            ]:  # FIXME: add more classes into elif
                dataset_dir = os.path.join("datasets", "MVTEC")
                if shot == "1":
                    dataset_dir = os.path.join(dataset_dir, "1shot")
                    fileids[cls] = {
                        **mvtec_matadata["1shot"]["mvtec_breakfast_box"],
                        **{"dataset_dir": dataset_dir},
                    }
                else:
                    # FIXME: create another shot options
                    raise Exception(
                        "other shots for MVTEC has not been implemented yet"
                    )
            else:
                with PathManager.open(
                    os.path.join(
                        split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                    )
                ) as f:
                    fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                    if isinstance(fileids_, str):
                        fileids_ = [fileids_]
                    fileids_ = [
                        fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                    ]  # TODO: list of image file ids without .jpg suffix
                    fileids[
                        cls
                    ] = fileids_  # TODO: dictionary, with "key" of classname
    else:
        with PathManager.open(
            os.path.join(dirname, "ImageSets", "Main", split + ".txt")
        ) as f:
            fileids = np.loadtxt(f, dtype=np.str)  # TODO: image file ids

    dicts = []
    if is_shots:
        # TODO: few shot fine tuning
        for cls, fileids_ in fileids.items():
            dicts_ = []  # dicts_ for each class
            if cls in [
                "nectarine",
                "orange",
                "cereal",
                "almond_mix",
            ]:  # FIXME: add more classes
                images, annotation, dataset_dir = (
                    fileids_["images"],
                    fileids_["annotation"],
                    fileids_["dataset_dir"],
                )
                anno_file = os.path.join(
                    dataset_dir, annotation
                )  # one anno file
                metadata = json.load(open(anno_file))
                for image in images:
                    image_file_path = os.path.join(dataset_dir, image)
                    image_anno = next(
                        d for d in metadata["files"] if d["name"] == image
                    )  # annotation for image
                    for obj in image_anno["objects"]:
                        if cls != obj["name"]:
                            continue
                        r = {
                            "file_name": image_file_path,
                            "image_id": image,  # FIXME: check if this causes trouble
                            "height": int(image_anno["size"]["height"]),
                            "width": int(image_anno["size"]["width"]),
                            "annotations": [
                                {
                                    "category_id": classnames.index(cls),
                                    "bbox": [float(x) for x in obj["bbox"]],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                }
                            ],
                        }
                        dicts_.append(r)
            else:
                for fileid in fileids_:
                    year = "2012" if "_" in fileid else "2007"
                    dirname = os.path.join("datasets", "VOC{}".format(year))
                    anno_file = os.path.join(
                        dirname, "Annotations", fileid + ".xml"
                    )
                    jpeg_file = os.path.join(
                        dirname, "JPEGImages", fileid + ".jpg"
                    )

                    tree = ET.parse(anno_file)

                    for obj in tree.findall("object"):
                        r = {
                            "file_name": jpeg_file,
                            "image_id": fileid,
                            "height": int(
                                tree.findall("./size/height")[0].text
                            ),
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
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

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


def register_meta_mvtec_style_voc(
    name, metadata, dirname, split, year, keepclasses, sid
):
    # TODO: split: one of train/test/val/trainval/novel_3shot_split_1_trainval(fine-tune), only used in base training
    # FIXME: let's keep "dirname" same as in pascal voc
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]

    # FIXME: check how DatasetCatalog and MetadataCatalog are utilized in repository
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
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
