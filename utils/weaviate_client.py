import weaviate
from weaviate.classes.config import Property, DataType, Configure
from utils.utils import resolve_config

config = resolve_config()
weaviate_config = config["weaviate"]

client = weaviate.connect_to_local(
    port=weaviate_config["port"], grpc_port=weaviate_config["grpc_port"]
)


def get_weaviate_client():
    return client


def get_or_create_class(client: weaviate.Client, class_name: str):
    if not class_name in client.collections.list_all().keys():
        if class_name == weaviate_config.get("papers_class_name"):
            collection = client.collections.create(
                name=class_name,
                properties=[
                    Property(name="arxiv_id", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="arxiv_url", data_type=DataType.TEXT),
                    Property(name="pdf_url", data_type=DataType.TEXT),
                    Property(name="published_date", data_type=DataType.DATE),
                    Property(name="abstract", data_type=DataType.TEXT),
                    Property(name="full_text", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            )
        else:
            raise ValueError(f"No default configuration for class {class_name}")
    else:
        collection = client.collections.get(class_name)
    return collection
