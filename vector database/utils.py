import base64
import os

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
    return encoded_image.decode("utf-8")

def get_image_urls_from_folder(folder_path: str):
    image_urls = []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(folder_path, filename)
            image_urls.append(image_path)

    return image_urls
