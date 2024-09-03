import configparser
import logging
import os
from operator import truediv

from scaleway import Client
from scaleway.inference.v1beta1.api import InferenceV1Beta1API
from scaleway.inference.v1beta1.types import EndpointSpec
from scaleway.inference.v1beta1.types import EndpointSpecPublic
from scaleway.serverless_sqldb.v1alpha1.api import ServerlessSqldbV1Alpha1API
from scaleway.instance.v1.api import InstanceV1API
from scaleway.rdb.v1.api import RdbV1API
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "mistral/mistral-7b-instruct-v0.3:bf16"
NODE_TYPE = "L4"
BUCKET_NAME = "test-cli"
DEPLOYMENT_NAME = "RAG"
INSTANCE_NAME = "RAG-instance"

logger = logging.getLogger('scaleway')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def deployment_exists():
    deployments = inference_api.list_deployments()
    for deployment_inference in deployments:
        if deployment_inference.name == DEPLOYMENT_NAME:
            return True
    return False

def instance_exists():
    instances = db_api.list_instances()
    for instance in instances:
        if instance.name == INSTANCE_NAME:
            return True
    return False

if __name__ == "__main__":
    client = Client.from_config_file_and_env("public", "~/.config/scw/config.yaml")
    inference_api = InferenceV1Beta1API(client)
    #db_api = ServerlessSqldbV1Alpha1API(client)
    #instance_api = InstanceV1API(client)
    db_api = RdbV1API(client)
    public_endpoint = EndpointSpecPublic()
    endpoint = EndpointSpec(
        disable_auth=False,
        public={},
        private_network=None,
    )
    project_id = os.getenv("PROJECT_ID", "")
    exist = deployment_exists()
    if not exist:
        deployment = inference_api.create_deployment(model_name=MODEL_NAME, node_type=NODE_TYPE, endpoints=[endpoint],
                                                 project_id=project_id, name=DEPLOYMENT_NAME)
    instance_exist = instance_exists()
    if not instance_exist:
        instance_db = db_api.create_instance(engine="PostgreSQL-15", user_name="laure-di", password="*********", node_type="DB-PLAY2-PICO", region="fr-par", is_ha_cluster=True,disable_backup=True, volume_size=10000000000000, volume_type="bssd", backup_same_region=True, name=INSTANCE_NAME)
        db = db_api.create_database(instance_id=instance_db.id, name=INSTANCE_NAME)
    endpoint_s3 = "https://s3." + os.getenv("DEFAULT_REGION", "") + ".scw.cloud"
    document = S3DirectoryLoader(bucket=BUCKET_NAME, endpoint_url=endpoint_s3,
                                 aws_access_key_id=os.getenv("ACCESS_KEY", ""),
                                 aws_secret_access_key=os.getenv("SECRET_KEY", ""))
    files = document.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(files)


