import json
from detectron2.data import Metadata
from detectron2.data import DatasetCatalog, MetadataCatalog

def load_metadata():
    # Load the metadata from the JSON file
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)

    # Create a Metadata object with the loaded metadata
    meta = Metadata()
    meta.thing_classes = metadata['thing_classes']

    return meta

def register_metadata():
    DatasetCatalog.register("my_dataset_train", lambda: None)
    MetadataCatalog.get("my_dataset_train").set(json.load(open('../data/model_config/metadata.json')))

if __name__ == "__main__":
    register_metadata()
