import supervision as sv

from roboflow import Roboflow
rf = Roboflow(api_key="WoJcpr1rwlUPyPJivcm2")

# grab our data
project = rf.workspace("cadyzedevelopmenthub").project("card-detection-zk7wu")
version = project.version(9)
dataset = version.download("yolov8")

# for each image, load YOLO annotations and require mask format for each
for subset in ["train", "test", "valid"]:
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/{subset}/images",
        annotations_directory_path=f"{dataset.location}/{subset}/labels",
        data_yaml_path=f"{dataset.location}/data.yaml",
        force_masks=True
    )
    ds.as_yolo(annotations_directory_path=f"{dataset.location}/{subset}/labels")