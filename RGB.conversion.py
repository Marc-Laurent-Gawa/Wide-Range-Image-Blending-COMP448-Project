import os
from PIL import Image

def convert_folder_to_rgb(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                img = Image.open(input_path).convert('RGB')
                img.save(output_path)
                print(f"Converted: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\Users\marcl\OneDrive\Documents\CONCORDIA\Summer 2024\COMP 353\Project\Wide-Range-Image-Blending-COMP448-Project\samples\input2"
    output_dir = None 
    convert_folder_to_rgb(input_dir, output_dir)