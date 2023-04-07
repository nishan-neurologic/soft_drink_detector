import os
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from config import MODEL_PATH, MODEL_NAME

def configure_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, MODEL_NAME)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def predict_image(cfg, image_path):
    predictor = DefaultPredictor(cfg)
    
    # Read and predict on the uploaded image
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Visualize the predictions
    train_metadata = MetadataCatalog.get("my_dataset_train") # Use train metadata instead of test metadata
    v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the detected results
    instances = outputs["instances"]
    pred_classes = instances.pred_classes
    pred_boxes = instances.pred_boxes.tensor.tolist()
    scores = instances.scores.tolist()

    for class_id, box, score in zip(pred_classes, pred_boxes, scores):
        class_name = train_metadata.thing_classes[class_id]
        print(f"Class: {class_name}, Confidence: {score:.2f}, Bounding Box: {box}")

def predict_image_flask(cfg, image_path):
    predictor = DefaultPredictor(cfg)
    
    # Read and predict on the uploaded image
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Visualize the predictions
    train_metadata = MetadataCatalog.get("my_dataset_train") # Use train metadata instead of test metadata
    v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print the detected results
    instances = outputs["instances"]
    pred_classes = instances.pred_classes
    pred_boxes = instances.pred_boxes.tensor.tolist()
    scores = instances.scores.tolist()
    output_image_path = os.path.splitext(image_path)[0] + '_predicted.jpg'
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
    for class_id, box, score in zip(pred_classes, pred_boxes, scores):
        class_name = train_metadata.thing_classes[class_id]
        print(f"Class: {class_name}, Confidence: {score:.2f}, Bounding Box: {box}")
        detected_objects = []
    for class_id, box, score in zip(pred_classes, pred_boxes, scores):
        class_name = train_metadata.thing_classes[class_id]
        detected_object = {"class": class_name, "confidence": score, "bounding_box": box}
        detected_objects.append(detected_object)

    return output_image_path, detected_objects  




if __name__ == "__main__":
    cfg = configure_predictor()
    image_path = "/Users/nishanali/WorkSpace/rani-peach/data/train/12_jpg.rf.b191224cf2de4ac4c514580b6676bbeb.jpg"
    predict_image(cfg, image_path)
