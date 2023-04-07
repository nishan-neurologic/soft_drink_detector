import os
import random
import cv2
import glob
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

import subprocess

import os
import urllib.request
import zipfile

from config import DATA_ROOT, MODEL_PATH, MODEL_NAME

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False)

def register_datasets():
    register_coco_instances("my_dataset_train", {}, f"{DATA_ROOT}/train/_annotations.coco.json", f"{DATA_ROOT}/train")
    register_coco_instances("my_dataset_val", {}, f"{DATA_ROOT}/valid/_annotations.coco.json", f"{DATA_ROOT}/valid")
    register_coco_instances("my_dataset_test", {}, f"{DATA_ROOT}/test/_annotations.coco.json", f"{DATA_ROOT}/test")

def configure_training():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.02

    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = (100, 150)
    cfg.SOLVER.GAMMA = 0.05
    cfg.OUTPUT_DIR = "./output"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = (50, 75)
    cfg.TEST.EVAL_PERIOD = 50
    cfg.MODEL.DEVICE = "cpu"

    return cfg

def train_model(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save the trained model
    torch.save(trainer.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))
    return trainer
def download_and_extract_data():
    # command = 'curl -L "https://app.roboflow.com/ds/KhjLZS7DRn?key=04wQwy2VHP" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip'
    # subprocess.run(command, shell=True, check=True)
    

    data_dir = "/Users/nishanali/WorkSpace/rani-peach/data"
    os.makedirs(data_dir, exist_ok=True)

    url = "https://app.roboflow.com/ds/KhjLZS7DRn?key=04wQwy2VHP"
    file_name = "roboflow.zip"
    file_path = os.path.join(data_dir, file_name)

    urllib.request.urlretrieve(url, file_path)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(file_path)
    return True

if __name__ == "__main__":
    _ = download_and_extract_data()
    register_datasets()
    cfg = configure_training()
    train_model(cfg)
