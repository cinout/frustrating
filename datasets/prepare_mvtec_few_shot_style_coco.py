import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 10], help="Range of seeds"
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = "datasets/cocosplit/datasplit/trainvalno5k.json"  # TODO: hard-coded file, found at http://dl.yf.io/fs-det/datasets/cocosplit/datasplit, default COCO annotation format
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data["categories"]:
        # TODO: "cat" example: {"supercategory": "person", "id": 1, "name": "person"}
        new_all_cats.append(cat)

    id2img = {}  # {img_id: img_meta}
    for i in data["images"]:
        # TODO: "i" example: {"license": 3, "file_name": "COCO_val2014_000000554625.jpg", "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000554625.jpg", "height": 640, "width": 426, "date_captured": "2013-11-14 16:03:19", "flickr_url": "http://farm5.staticflickr.com/4086/5094162993_8f59d8a473_z.jpg", "id": 554625}
        id2img[i["id"]] = i

    anno = {i: [] for i in ID2CLASS.keys()}  # {class_id: object_annotation}
    for a in data["annotations"]:
        # TODO: "a" example: {"segmentation": [[263.59, ..., 206.92]], "area": 13203.3917, "iscrowd": 0, "image_id": 194306, "bbox": [232.68, 103.89, 183.73, 103.03], "category_id": 18, "id": 2496}
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    random.seed(0)
    for c in ID2CLASS.keys():  # c is class_id
        img_ids = (
            {}
        )  # TODO: {img_id: anno[]} for class c, anno contains that img_id
        for a in anno[c]:
            if a["image_id"] in img_ids:
                img_ids[a["image_id"]].append(a)
            else:
                img_ids[a["image_id"]] = [a]

        sample_shots = []  # anno[]
        sample_imgs = []  # img_meta[]
        for shots in [1, 2, 3, 5]:  # FIXME: change later [1, 2, 3, 5, 10, 30]
            while True:
                imgs = random.sample(list(img_ids.keys()), shots)
                for img in imgs:
                    skip = False
                    for s in sample_shots:
                        if img == s["image_id"]:
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_ids[img]) + len(sample_shots) > shots:
                        continue
                    sample_shots.extend(img_ids[img])
                    sample_imgs.append(id2img[img])
                    if len(sample_shots) == shots:
                        break
                if len(sample_shots) == shots:
                    break
            new_data = {
                "info": data["info"],
                "licenses": data["licenses"],
                "images": sample_imgs,
                "annotations": sample_shots,
            }
            save_path = get_save_path_seeds(ID2CLASS[c], shots)
            new_data["categories"] = new_all_cats
            with open(save_path, "w") as f:
                json.dump(new_data, f)


def get_save_path_seeds(cls, shots):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join("datasets", "mvtecsplit_cocostyle")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }  # FIXME: add our classes
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}  # TODO: {name: id}

    args = parse_args()
    generate_seeds(args)
