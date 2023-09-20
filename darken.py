import cv2
import numpy as np
import random
import os

def reduce_brightness(input_image_path, output_image_path, brightness_range=(0.4, 0.6)):
    """
    Reduces the brightness of an image and saves it.

    Parameters:
    - input_image_path (str): The path to the input image.
    - output_image_path (str): The path to save the output image.
    - brightness_range (tuple): A tuple specifying the brightness reduction range (min, max).

    Returns:
    - None
    """
    # Read the image
    img = cv2.imread(input_image_path)

    # Generate a random factor within the specified brightness range
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])

    # Apply brightness reduction
    img_brightness_reduced = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)

    # Create the output directory if it doesn't exist
    result_dir = os.path.dirname(output_image_path)
    os.makedirs(result_dir, exist_ok=True)

    # Save the resulting image
    cv2.imwrite(output_image_path, img_brightness_reduced, [cv2.IMWRITE_PNG_COMPRESSION, 9])

if __name__ == "__main__":
    input_image_path = "plasmodium-phone-0001.jpg"  # Replace with your input image path
    output_image_path = "data/result/plsmo.png"  # Specify the desired output image path

    reduce_brightness(input_image_path, output_image_path)
    print(f"Brightness reduced and saved to '{output_image_path}'")
