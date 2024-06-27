from ultralytics import YOLO

def train():

    # Load a model
    model = YOLO("runs/detect/train40/weights/best.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="datasets/Card-Detection-8/data.yaml", epochs=3, imgsz=640, device=0, cache="disk", batch=1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    path = model.export(format="onnx")  # export the model to ONNX format


def validate():
    model = YOLO("runs/detect/train44/weights/best.pt")
    metrics = model.val()
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps 

if __name__ == '__main__':
    validate()