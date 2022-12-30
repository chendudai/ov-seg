# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config
from pathlib import Path

from open_vocab_seg.utils import VisualizationDemo
import fnmatch
import json

# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    relevant_paths = []


    with open('/storage/chendudai/data/test_undistorted/names.txt') as f:
        lines_names = f.readlines()
        for line in lines_names:
            if line == '\n':
                continue
            relevant_paths.append(line.split('\t')[1].split('\n')[0])


    path = '/storage/chendudai/data/WikiScenes/cathedrals/0/'
    captions = []
    path_images = []
    cnt = 0
    long_captions = []
    list_categories = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.json'):
            path_json = os.path.join(root, filename)

            with open(path_json, 'r', encoding='utf-8') as f:
                my_data = json.load(f)
            for image in my_data['pictures']:
                caption = my_data['pictures'][image]['caption']
                file_path = os.path.join(path_json[:-13],'pictures', image)
                captions.append(caption)
                path_images.append(file_path)

                # Extract Categories For The Images
                categories = []
                path_categories = Path(file_path)
                while path_categories.name != 'cathedrals':
                    son_folder = path_categories.name
                    path_categories = path_categories.parent
                    if path_categories.name == 'pictures':
                        path_categories = path_categories.parent
                        continue

                    for filename_category in fnmatch.filter(filenames, '*.json'):
                        path_json_category = os.path.join(path_categories, filename_category)
                        with open(path_json_category, 'r', encoding='utf-8') as f:
                            folder_data = json.load(f)

                        category = list(folder_data['pairs'].keys())[int(son_folder)]
                        categories.append(category)

                list_categories.append(categories)

    print("len[captions] = ",len(captions))

    relevant_categories = []
    for p_rel in relevant_paths:
        for i, p in enumerate(path_images):
            if p[46:] == p_rel:
                relevant_categories.append(list_categories[i][0])
                break
            if i == len(path_images) - 1:
                relevant_paths.remove(p_rel)


    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            i = 0
            # for filename in os.listdir(path):
            for j, filename in enumerate(relevant_paths):
                print(i)
                i = i + 1

                f = os.path.join(path, filename)
                if os.path.isfile(f):

                    img = read_image(f, format="BGR")
                    start_time = time.time()
                    class_names = [relevant_categories[j],'others']
                    predictions, visualized_output = demo.run_on_image(img, class_names)
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            f,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            out_filename = os.path.join(args.output, os.path.basename(f))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                        visualized_output.save(out_filename)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
    else:
        raise NotImplementedError