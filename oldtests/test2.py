import cv2
from pytesseract import pytesseract, Output

# Specify Tesseract command path
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load an image
image = cv2.imread('nums.jpg')  # Update with your image path
if image is None:
    print("Image not found")
else:
    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(rgb)
    print("Recognized Text:", text)
