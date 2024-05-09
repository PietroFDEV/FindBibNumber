import cv2
import numpy as np
from pytesseract import pytesseract, Output

# Specify the path to the Tesseract executable
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    image = cv2.resize(img, None, fx=0.4, fy=0.4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return image, sorted_ctrs

def recognize_characters(image, contours):
    results = []
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 2 and h > 2:
            roi = image[y:y+h, x:x+w]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb, config='--psm 10 --oem 3')  # Adjust psm and oem based on your requirements
            results.append((text, (x, y, w, h)))
    return results

def main(image_path):
    image, contours = preprocess_image(image_path)
    recognized_data = recognize_characters(image, contours)
    
    # Displaying the results
    for text, (x, y, w, h) in recognized_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Using OpenCV's imshow to display the image with recognized text
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_your_image.jpg' with your actual image path
main('nums.jpg')
