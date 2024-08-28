import configparser
import logging
import os

from scaleway import Client
from scaleway.inference.v1beta1.api import InferenceV1Beta1API
from scaleway.inference.v1beta1.types import EndpointSpec
from scaleway.inference.v1beta1.types import EndpointSpecPublic
from scaleway.serverless_sqldb.v1alpha1.api import ServerlessSqldbV1Alpha1API
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "mistral/mistral-7b-instruct-v0.3:bf16"
NODE_TYPE = "L4"
BUCKET_NAME = "test-cli"


logger = logging.getLogger('scaleway')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def _create_client_from_env():
    client_from_env = Client(
        access_key=os.getenv("ACCESS_KEY", ""),
        secret_key=os.getenv("SECRET_KEY", ""),
        default_project_id=os.getenv("PROJECT_ID", ""),
        default_region=os.getenv("DEFAULT_REGION", ""),
        default_zone=os.getenv("DEFAULT_ZONE", ""),
    )
    return client_from_env


if __name__ == "__main__":
    client = _create_client_from_env()
    inference_api = InferenceV1Beta1API(client)
    db_api = ServerlessSqldbV1Alpha1API(client)
    public_endpoint = EndpointSpecPublic()
    endpoint = EndpointSpec(
        disable_auth=False,
        public={},
        private_network=None,
    )
    project_id = os.getenv("PROJECT_ID", "")
    deployment = inference_api.create_deployment(model_name=MODEL_NAME, node_type=NODE_TYPE, endpoints=[endpoint],
                                                 project_id=project_id)
    database = db_api.create_database(name="RAG", cpu_min=2, cpu_max=15, project_id=project_id)
    endpoint_s3 = "https://s3." + os.getenv("DEFAULT_REGION", "") + ".scw.cloud"
    document = S3DirectoryLoader(bucket=BUCKET_NAME, endpoint_url=endpoint_s3,
                                 aws_access_key_id=os.getenv("ACCESS_KEY", ""),
                                 aws_secret_access_key=os.getenv("SECRET_KEY", ""))
    files = document.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(files)


