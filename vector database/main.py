from titan_embedder import TitanEmbedder
from faiss_vector_db import FaissVectorDB
from utils import get_image_urls_from_folder
from PIL import Image

def show_images(image_paths):
    for path in image_paths:
        try:
            img = Image.open(path)
            img.show()  # mở ảnh bằng trình xem ảnh mặc định
        except Exception as e:
            print(f"Không mở được ảnh {path}: {e}")

def main():
    print("===== START PROGRAM =====")

    # 1. Tạo TitanEmbedder (mock)
    embedder = TitanEmbedder(dim=512)

    # 2. Lấy danh sách ảnh
    image_urls = get_image_urls_from_folder("images")
    print("Image URLs:", image_urls)

    # 3. Tạo embedding cho từng ảnh
    image_embeddings = [embedder.embed_image(url) for url in image_urls]

    # 4. Tạo FAISS database
    db = FaissVectorDB(image_urls, image_embeddings, k=3)
from titan_embedder import TitanEmbedder
from faiss_vector_db import FaissVectorDB
from utils import get_image_urls_from_folder
from PIL import Image

def show_images(image_paths):
    for path in image_paths:
        try:
            img = Image.open(path)
            img.show()
        except Exception as e:
            print(f"Không mở được ảnh {path}: {e}")

def main():
    embedder = TitanEmbedder(dim=1024)  # dùng Titan thật
    image_urls = get_image_urls_from_folder("images")
    image_embeddings = [embedder.embed_image(url) for url in image_urls]
    db = FaissVectorDB(image_urls, image_embeddings, k=3)

    query_text = input("Nhập thông tin cần tìm kiếm: ")
    query_embedding = embedder.embed_text(query_text)
    results = db.search(query_embedding)
    print("Kết quả tìm kiếm:", results)
    show_images(results)

if __name__ == "__main__":
    main()

    # 5. Nhận thông tin cần tìm kiếm (ví dụ text)
    query_text = input("Nhập thông tin cần tìm kiếm: ")
    query_embedding = embedder.embed_text(query_text)

    # 6. Tìm kiếm và in ra ảnh phù hợp
    results = db.search(query_embedding)
    print("Kết quả tìm kiếm:", results)
    show_images(results)

if __name__ == "__main__":
    main()
