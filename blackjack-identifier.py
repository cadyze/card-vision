from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("runs/detect/train40/weights/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["datasets/Card-Detection-8/test/images/021407554_jpg.rf.d68568480c4c6d78df79181201aebfe0.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen

def preprocess_image(image, input_shape):
    # Resize image to the model's input size
    # input_image = cv2.resize(image, input_shape)
    # Convert image to RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # Normalize and convert to tensor
    input_image = input_image / 255.0
    input_image = np.transpose(input_image, (2, 0, 1)).astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame, stream=True)
    for result in results:
        cv2.imshow("Card-Vision", result.plot())
    
    # cv2.imshow("ffff", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()