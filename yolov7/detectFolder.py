import os
from subprocess import run

def process_images(image_folder, script_path, base_params):

    # Check if image folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        return

    # Get image filenames (ensure they are image files)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    if not image_files:
        print(f"Warning: No image files found in '{image_folder}'.")
        return

    # Loop through images, constructing full source paths and running the script
    for image_name in image_files:
        full_source = os.path.join(image_folder, image_name)

        # Construct command with specific source path and base parameters
        command = ["python", script_path]
        for key, value in base_params.items():
          command.extend([f"--{key}", str(value)])
        command.extend(["--source", full_source])

        print(f"Executing: {' '.join(command)}")  # Show the command being run
        run(command)

# Example usage
image_folder = "../photos/"
script_path = "detect.py"
base_params = {
    "weights": "runs/train/yolov7-custom/weights/best.pt",
    "conf": 0.12,
    "img-size": 640,
    "device": "cpu",
    "save-txt": "--no-trace",
}

process_images(image_folder, script_path, base_params)
