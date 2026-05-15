import os
from PIL import Image, ImageOps

def invert_image_in_place(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            inverted = ImageOps.invert(img)
            inverted.save(path)
        print(f"Inverted: {path}")
    except Exception as e:
        print(f"Failed: {path} -> {e}")

def is_image_file(path):
    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    return os.path.isfile(path) and path.lower().endswith(supported_ext)

if __name__ == "__main__":
    print("Image inverter (in-place)")
    print("Enter image paths one by one.")
    print("Type 'exit' to stop.\n")

    while True:
        input_path = input("Image path: ").strip().strip('"')

        if input_path.lower() == "exit":
            print("Stopping.")
            break

        if not os.path.exists(input_path):
            print("Path does not exist.\n")
            continue

        if not is_image_file(input_path):
            print("Not a supported image file.\n")
            continue

        invert_image_in_place(input_path)