import boto3
import json
import base64
import numpy as np
import numpy.typing as npt


class TitanEmbedder:

    def __init__(
        self,
        region: str = "us-east-1",
        dim: int = 1024
    ) -> None:

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region
        )

        self.dim = dim

        print("TitanEmbedder (AWS Bedrock) created")


    def embed_image(
        self,
        image_path: str
    ) -> npt.NDArray:

        print("Embedding image:", image_path)

        with open(image_path, "rb") as f:

            image_bytes = f.read()


        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")


        body = json.dumps({

            "inputImage": image_base64,

            "embeddingConfig": {

                "outputEmbeddingLength": self.dim

            }

        })


        response = self.client.invoke_model(

            modelId="amazon.titan-embed-image-v1",

            body=body,

            contentType="application/json",

            accept="application/json"

        )


        result = json.loads(
            response["body"].read()
        )


        embedding = result["embedding"]


        print(
            "Embedding dimension:",
            len(embedding)
        )


        return np.array(
            embedding,
            dtype="float32"
        )


    def embed_text(
        self,
        text: str
    ) -> npt.NDArray:


        body = json.dumps({

            "inputText": text,

            "embeddingConfig": {

                "outputEmbeddingLength": self.dim

            }

        })


        response = self.client.invoke_model(

            modelId="amazon.titan-embed-image-v1",

            body=body,

            contentType="application/json",

            accept="application/json"

        )


        result = json.loads(
            response["body"].read()
        )


        embedding = result["embedding"]


        return np.array(
            embedding,
            dtype="float32"
        )