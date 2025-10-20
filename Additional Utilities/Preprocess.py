import os
from PIL import Image, ImageOps
import numpy as np

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
                img = ImageOps.exif_transpose(img)  # fix rotation issues

                # Convert to NumPy array
                img_np = np.array(img, dtype=np.float32) / 255.0

                # Clip any HDR overflow
                img_np = np.clip(img_np, 0.0, 1.0)

                # Apply gamma correction (fix high contrast panoramas)
                img_np = np.power(img_np, 1/2.2)

                # Normalize to [-1, 1] for training stability
                img_np = (img_np - 0.5) / 0.5

                # Convert back to [0, 255] uint8 for saving
                img_out = ((img_np + 1) / 2.0 * 255.0).astype(np.uint8)
                img_out = Image.fromarray(img_out)

                # Save processed image
                img_out.save(output_path)
                print(f"Converted: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\Users\marcl\Downloads\CVRG-Pano\train\rgb"
    output_dir = None 
    convert_folder_to_rgb(input_dir, output_dir)