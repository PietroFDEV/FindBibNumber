import torch
import cv2
from PIL import Image
import pytesseract
import numpy as np

# Load the trained YOLOv7 model weights
model = torch.hub.load('C:/Users/lucas/OneDrive/Documentos/GitHub/NeuralNetwork/yolov7', 'custom', 'runs/train/yolov7-custom/weights/last.pt', source='local')

device = torch.device('cuda')
model.to(device)

def detect_runners(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    results = model(img, size=640)  # Ensure size is appropriate for your model
    results.save('output/')  # Save the results to the output folder
    
    # Debug: Print results
    print("Detection results:", results.xyxy)
    
    return results

# def read_numbers(image_path, detections):
#     img = cv2.imread(image_path)
#     number_list = []
#     for bbox in detections.xyxy[0]:  # Assuming detections are in xyxy format
#         x1, y1, x2, y2, conf, cls = bbox
#         cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
#         gray_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
#         _, bw_cropped_img = cv2.threshold(gray_cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         number = pytesseract.image_to_string(bw_cropped_img, config='--psm 8')
#         number_list.append(number.strip())
#     return number_list

def draw_detections(image_path, detections, conf_threshold=0.25):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    for bbox in detections.xyxy[0]:  # Assuming detections are in xyxy format
        x1, y1, x2, y2, conf, cls = bbox[:6]
        if conf > conf_threshold:  # Only process boxes above confidence threshold
            # Convert from xyxy to YOLO format
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label and confidence score
            label = f'Class {int(cls)}: {conf:.2f}'
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Debug: Print YOLO format values
            print(f"Class: {int(cls)}, Center X: {center_x}, Center Y: {center_y}, Width: {width}, Height: {height}")
        
    output_path = 'output/detections.jpg'
    cv2.imwrite(output_path, img)
    print(f"Detections saved to {output_path}")
    return img

# Example usage
image_path = 'vanderlei.jpg'
detections = detect_runners(image_path)
drawn_image = draw_detections(image_path, detections)
# numbers = read_numbers(image_path, detections)
# print(numbers)
