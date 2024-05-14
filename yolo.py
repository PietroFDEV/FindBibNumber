import cv2

# Load YOLO model
def load_yolo_model():
    yolo_model = cv2.dnn.readNet("yolov3.weights", "cfg/yolov3.cfg")
    return yolo_model

# Detect objects using YOLO
def detect_objects_yolo(image, yolo_model):
    # Get class labels
    with open("cfg/coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Extract output layers names
    layer_names = yolo_model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_model.setInput(blob)
    outputs = yolo_model.forward(output_layers)

    # Get bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Extract bounding boxes of detected objects
    bounding_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

# Main function
def main():
    # Load YOLO model
    yolo_model = load_yolo_model()

    # Load and process image
    image_path = 'data/dog.jpg'
    image = cv2.imread(image_path)

    # Detect objects using YOLO
    bounding_boxes = detect_objects_yolo(image, yolo_model)

    print("Bounding boxes:", bounding_boxes)

if __name__ == "__main__":
    main()
