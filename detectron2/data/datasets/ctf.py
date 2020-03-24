# author: chenwr 
# datetime: 20191210
# function: transform suakit json to detectron2 dataset format

import json
import os
import glob
import logging
import numpy as np

from PIL import Image
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager
from itertools import chain

from detectron2.data.datasets.ctf_labels import labels, labels_ctf, labels_gold, labels_999

try:
    import cv2
except ImportError:
    pass



CLASS_NAMES_FIELD = [
    "ctf", "gold", "999", "other",
]

CLASS_NAMES_CHAR = [
    "c", "t", "f"
]




def load_ctf_json(image_dir, gt_dir, dataset_name):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/datasets/ctf/char/train".
        gt_dir (str): path to the raw json. e.g., "~/datasets/ctf/char/json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing ctf {} annotations ...".format(dataset_name))

    
    if "field" in dataset_name:
        dataset_dicts = field_to_dict(image_dir,gt_dir)
    elif "char" in dataset_name:
        dataset_dicts = char_to_dict(image_dir,gt_dir)

    logger.info("Loaded {} images from {}".format(len(dataset_dicts), image_dir))
    return dataset_dicts


def field_to_dict(image_dir,gt_dir):
    """
    Parse cityscapes annotation files to a dict.

    Args:
        image_dir (str): path to the raw dataset. e.g., "~/datasets/ctf/char/train".
        gt_dir (str): path to the raw json. e.g., "~/datasets/ctf/char/json".

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from shapely.geometry import Polygon

    dataset_dicts=[]
    image_id=0
    for image_file in glob.glob(os.path.join(image_dir, "*.jpg")):
        suffix=".jpg"
        assert image_file.endswith(suffix)
        prefix=image_dir
        assert image_file.startswith(prefix)
        json_file=gt_dir + image_file[len(prefix):-len(suffix)] + "_label.json"
        assert "No image found in {}".format(image_file)
        assert "No json found in {}".format(json_file)

        with PathManager.open(image_file,"rb") as f:
            inst_image=np.asarray(Image.open(f),order="F")
            record={
                "file_name":image_file,
                "image_id":image_id,
                "height":inst_image.shape[0], 
                "width":inst_image.shape[1]
            }

        with PathManager.open(json_file,"r") as f:
            jsonobj=json.load(f)
        
        annos=[]
        for obj in jsonobj["Labels"]:
            assert obj["Shape"]=="rectangle"
            anno={}

            label=obj["Class"]
            if label < 0:
                continue
            anno["category_id"]=label
            # segmentation
            points= np.array(list(map(eval, obj["Points"])), dtype=float)
            xmin, ymin =np.min(points,axis=0)
            xmax, ymax =np.max(points,axis=0)
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            points_list=np.array([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax, ymin]])
            segm= [list(np.asarray(points_list).flatten())]
            if segm:  # either list[list[float]] or dict(RLE)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    continue  # ignore this instance
            anno["segmentation"] = segm


            annos.append(anno)

        record["annotations"] = annos
        dataset_dicts.append(record)
        image_id +=1

    return dataset_dicts



def char_to_dict(image_dir,gt_dir):
    """
    Parse cityscapes annotation files to a dict.

    Args:
        image_dir (str): path to the raw dataset. e.g., "~/datasets/ctf/char/train".
        gt_dir (str): path to the raw json. e.g., "~/datasets/ctf/char/json".

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from shapely.geometry import Polygon

    dataset_dicts=[]
    image_id=0
    for image_file in glob.glob(os.path.join(image_dir, "*.jpg")):
        suffix = ".jpg"
        assert image_file.endswith(suffix)
        prefix = image_dir
        json_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "_label.json"
        assert len(image_file), "No images found in {}".format(image_dir)
        assert len(json_file), "No json found in {}".format(json_file)

        with PathManager.open(image_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        record = {
            "file_name": image_file,
            "image_id": image_id,
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        annos = []
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)

        # ctf char label draw the polygons in sequential order
        for obj in jsonobj["Labels"][::-1]:
            # label = obj["Class"]+4
            label = obj["Class"]

            # label转换为label_name
            # try:
            #     label = CLASS_NAMES_FIELD.index(label_name)
            # except KeyError:
            #     print("key error")

            if label < 0:
                continue

            anno = {}
            anno["category_id"] = label
            # segmentation
            points= np.array(list(map(eval, obj["Points"])), dtype=float)
            segm= [list(np.asarray(points).flatten())]
            if segm:  # either list[list[float]] or dict(RLE)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    continue  # ignore this instance
            anno["segmentation"] = segm

            (xmin, ymin, xmax, ymax) = Polygon(points).bounds
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(anno)

        record["annotations"] = annos
        dataset_dicts.append(record)
        image_id +=1
    return dataset_dicts



if __name__ == '__main__':
    """
    Test the ctf dataset loader. 
    
    Usage:

    """

    """
    Test the ctf dataset loader.

    Usage:
        python -m detectron2.data.datasets.ctf \
            datasets/ctf/char/train datasets/ctf/char/json
    """
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir",type=str, default="datasets/ctf/field/train")
    # parser.add_argument("--gt_dir",type=str, default="datasets/ctf/field/json-field")
    parser.add_argument("--image_dir",type=str, default="datasets/ctf/char/train")
    parser.add_argument("--gt_dir",type=str, default="datasets/ctf/char/json")

    parser.add_argument("--type", choices=["field", "char"], default="char")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)

    dirname = "ctf-data-vis"
    os.makedirs(dirname, exist_ok=True)
    dicts = load_ctf_json(args.image_dir, args.gt_dir, args.type)
    logger.info("Done loading {} samples.".format(len(dicts)))

    thing_classes = [k.name for k in labels_ctf]
    meta = Metadata().set(thing_classes=thing_classes)
    
    # stuff_names = [k.name for k in labels]
    # stuff_colors = [k.color for k in labels]
    # meta = Metadata().set(stuff_names=stuff_names, stuff_colors=stuff_colors)

    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img,metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("a", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)




