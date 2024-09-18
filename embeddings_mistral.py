import logging
import os


import psycopg2
from scaleway import Client
from scaleway.inference.v1beta1.api import InferenceV1Beta1API
from scaleway.inference.v1beta1.types import EndpointSpec
from scaleway.inference.v1beta1.types import EndpointSpecPublic
from scaleway.rdb.v1.api import RdbV1API
from dotenv import load_dotenv
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.testing.plugin.plugin_base import engines

MODEL_NAME = "mistral/mistral-7b-instruct-v0.3:bf16"
NODE_TYPE = "L4"
BUCKET_NAME = "cli-test"
DEPLOYMENT_NAME = "RAG"
INSTANCE_NAME = "RAG-instance"
DB_NAME = "DB_RAG"
DB_USER = "laure-di"

logger = logging.getLogger('scaleway')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

load_dotenv()

def deployment_exists():
    deployments = inference_api.list_deployments()
    if deployments.total_count == 0:
        return False
    for deployment_inference in deployments.deployments:
        if deployment_inference.name == DEPLOYMENT_NAME:
            return True
    return False

def instance_exists():
    instances = db_api.list_instances()
    if instances.total_count == 0:
        return False
    for instance in instances.instances:
        if instance.name == INSTANCE_NAME:
            return True
    return False

def deployment_by_name(deployment_name):
    deployments = inference_api.list_deployments()
    for deployment_inference in deployments.deployments:
        if deployment_inference.name == deployment_name:
            return deployment_inference

def instance_by_name(image_name):
    instances = db_api.list_instances()
    for instance in instances.instances:
        if instance.name == INSTANCE_NAME:
            return instance

def db_by_instance(instance_id):
    databases = db_api.list_databases(instance_id=instance_id)
    for database in databases.databases:
        if database.name == DB_NAME:
            return database

if __name__ == "__main__":
    client = Client.from_config_file_and_env("/Users/lmasson/.config/scw/config.yaml", "public")
    inference_api = InferenceV1Beta1API(client)
    db_api = RdbV1API(client)
    public_endpoint = EndpointSpecPublic()
    endpoint = EndpointSpec(
        disable_auth=False,
        public={},
        private_network=None,
    )
    project_id = os.getenv("SCW_DEFAULT_PROJECT_ID", "")
    exist = deployment_exists()
    if not exist:
        deployment = inference_api.create_deployment(model_name=MODEL_NAME, node_type=NODE_TYPE, endpoints=[endpoint],
                                                 project_id=project_id, name=DEPLOYMENT_NAME)
    else:
        deployment = deployment_by_name(DEPLOYMENT_NAME)


    instance_exist = instance_exists()
    if not instance_exist:
        instance_db = db_api.create_instance(engine="PostgreSQL-15", user_name="laure-di", password=os.getenv("SCW_DB_PASSWORD"), node_type="db-play2-nano", region="fr-par", is_ha_cluster=False,disable_backup=True, volume_size=10000000000, volume_type="bssd", backup_same_region=False, name=INSTANCE_NAME)
        db_api.wait_for_instance(instance_id=instance_db.id)
        db = db_api.create_database(instance_id=instance_db.id, name=DB_NAME)

    else:
        instance_db = instance_by_name(INSTANCE_NAME)
        db = db_by_instance(instance_db.id)
    logger.debug("start trying to connect to database")
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=os.getenv("SCW_DB_PASSWORD"),
        host=instance_db.endpoint.ip,  # Use the public/private IP or hostname
        port=instance_db.endpoint.port  # Use the correct port
    )
    cur = conn.cursor()
    logger.debug("connected to database")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_vectors (
            id SERIAL PRIMARY KEY,
            document_id TEXT,
            content TEXT,
            vector VECTOR(1536)  -- Adjust the dimension based on your embeddings
        );
    """)
    logger.debug("finish request")
    conn.commit()
    # logger.debug("start loading documents")
    # endpoint_s3 = "https://s3." + os.getenv("SCW_DEFAULT_REGION", "") + ".scw.cloud"
    # document = S3DirectoryLoader(bucket=BUCKET_NAME, endpoint_url=endpoint_s3,
    #                              aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
    #                              aws_secret_access_key=os.getenv("SCW_SECRET_KEY", ""))
    # files = document.load()
    # logger.debug("finished loading documents")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # logger.debug("splitting in chunks")
    # chunks = text_splitter.split_documents(files)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("SCW_API_KEY"), openai_api_base=os.getenv("ENDPOINT"), model="mistral/mistral-7b-instruct-v0.3:bf16")
    # for chunk in chunks:
    query_string = (
        "PostgreSQL is my favorite database"
    )
    text2 = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs"
    )
    if not query_string:
        logger.debug("something is wrong with string")
    else:
        embed = embeddings.embed_documents([query_string, text2])
        print(len(embed))  # Should be 1536, the dimensionality of OpenAI embeddings
        print(embed[:5])
    # cur.execute("INSERT INTO document_vectors (document_id, content, vector) VALUES (%s, %s, %s)", (chunk.metadata.get('id', 'unknown_id'), chunk.page_content, vector))
    # conn.commit()
