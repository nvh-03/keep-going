import numpy as np
import numpy.typing as npt
import faiss
from typing import List

class FaissVectorDB:
    def __init__(
        self,
        image_urls: List[str],
        image_embeddings: List[npt.NDArray],
        k: int = 3
    ) -> None:

        embeddings_array = np.array(image_embeddings).astype("float32")
        print("Embeddings shape:", embeddings_array.shape)

        self.index_to_image = {i: image for i, image in enumerate(image_urls)}

        d = embeddings_array.shape[1]
        print("Embedding dimension:", d)

        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings_array)

        print("FAISS vectors:", self.index.ntotal)

        self.k = k

    def search(self, query_embedding: npt.NDArray) -> List[str]:
        query_embedding_2d = np.array([query_embedding]).astype("float32")

        distances, result_indices = self.index.search(query_embedding_2d, self.k)
        print("Distances:", distances)

        return [self.index_to_image[i] for i in result_indices[0]]
