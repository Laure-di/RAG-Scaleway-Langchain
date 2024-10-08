import os

import boto3
import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import S3FileLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scaleway.inference.v1beta1 import InferenceV1Beta1API, EndpointSpecPublic, EndpointSpec
from scaleway.rdb.v1 import RdbV1API, Permission
from scaleway_core.client import Client

MODEL_NAME = "mistral/mistral-7b-instruct-v0.3:bf16"
NODE_TYPE = "L4"
BUCKET_NAME = "rag-test"
DEPLOYMENT_NAME = "RAG"
INSTANCE_NAME = "RAG-instance"
DB_NAME = "DB_RAG"
DB_USER = "laure-di"
MODEL_NAME_EMBED = "sentence-transformers/sentence-t5-xxl:fp32"
DEPLOYMENT_NAME_EMBED = "RAG-embeddings"

# Load environment variables
load_dotenv()


def deployment_exists(deployment_name, inference_api):
    deployments = inference_api.list_deployments()
    return any(deployment.name == deployment_name for deployment in deployments.deployments)


def instance_exists(db_api):
    instances = db_api.list_instances()
    return any(instance.name == INSTANCE_NAME for instance in instances.instances)


def deployment_by_name(deployment_name, inference_api):
    deployments = inference_api.list_deployments()
    for deployment in deployments.deployments:
        if deployment.name == deployment_name:
            return deployment


def instance_by_name(db_api):
    instances = db_api.list_instances()
    for instance in instances.instances:
        if instance.name == INSTANCE_NAME:
            return instance


def db_by_instance(instance_id, db_api):
    databases = db_api.list_databases(instance_id=instance_id)
    for database in databases.databases:
        if database.name == DB_NAME:
            return database


def setup_rag():
    client = Client.from_config_file_and_env("/Users/lmasson/.config/scw/config.yaml", "public")
    inference_api = InferenceV1Beta1API(client)
    db_api = RdbV1API(client)
    public_endpoint = EndpointSpecPublic()
    endpoint = EndpointSpec(disable_auth=False, public=public_endpoint, private_network=None)
    project_id = os.getenv("SCW_DEFAULT_PROJECT_ID", "")
    # Deployment for LLM
    if not deployment_exists(DEPLOYMENT_NAME, inference_api):
        deployment = inference_api.create_deployment(
            model_name=MODEL_NAME,
            node_type=NODE_TYPE,
            endpoints=[endpoint],
            project_id=project_id,
            name=DEPLOYMENT_NAME
        )
    else:
        deployment = deployment_by_name(DEPLOYMENT_NAME, inference_api)
    # Deployment for embeddings
    if not deployment_exists(DEPLOYMENT_NAME_EMBED, inference_api):
        deployment_embeddings = inference_api.create_deployment(
            model_name=MODEL_NAME_EMBED,
            node_type=NODE_TYPE,
            endpoints=[endpoint],
            project_id=project_id,
            name=DEPLOYMENT_NAME_EMBED
        )
    else:
        deployment_embeddings = deployment_by_name(DEPLOYMENT_NAME_EMBED, inference_api)
    # logger.debug("Create or get DB")
    if not instance_exists(db_api):
        instance_db = db_api.create_instance(
            engine="PostgreSQL-15",
            user_name=DB_USER,
            password=os.getenv("SCW_DB_PASSWORD"),
            node_type="db-play2-nano",
            region="fr-par",
            is_ha_cluster=False,
            disable_backup=True,
            volume_size=10000000000,
            volume_type="bssd",
            backup_same_region=False,
            name=INSTANCE_NAME
        )
        instance_db = db_api.wait_for_instance(instance_id=instance_db.id)
        db_api.create_database(instance_id=instance_db.id, name=DB_NAME)
        db_api.set_privilege(instance_id=instance_db.id, database_name=DB_NAME, user_name=DB_USER,
                             permission=Permission.ALL)
    else:
        instance_db = instance_by_name(db_api)
        db_by_instance(instance_db.id, db_api)

    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=os.getenv("SCW_DB_PASSWORD"),
        host=instance_db.endpoint.ip,
        port=instance_db.endpoint.port
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cur.execute("CREATE TABLE IF NOT EXISTS object_loaded (id SERIAL PRIMARY KEY, object_key TEXT)")
    conn.commit()
    embedding_url = deployment_embeddings.endpoints[0].url + "/v1"
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("SCW_SECRET_KEY"),
        openai_api_base=embedding_url,
        model="sentence-transformers/sentence-t5-xxl",
        tiktoken_enabled=False,
    )
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    connection_string = f"postgresql+psycopg2://{conn.info.user}:{conn.info.password}@{conn.info.host}:{conn.info.port}/{conn.info.dbname}"
    vector_store = PGVector(
        connection=connection_string,
        embeddings=embeddings,
    )
    endpoint_s3 = f"https://s3.{os.getenv('SCW_DEFAULT_REGION', '')}.scw.cloud"
    session = boto3.session.Session()
    client_s3 = session.client(service_name='s3', endpoint_url=endpoint_s3,
                               aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
                               aws_secret_access_key=os.getenv("SCW_SECRET_KEY", ""))
    paginator = client_s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True,
                                                   length_function=len, is_separator_regex=False)
    for page in page_iterator:
        for obj in page.get('Contents', []):
            cur.execute("SELECT object_key FROM object_loaded WHERE object_key = %s", (obj['Key'],))
            response = cur.fetchone()
            if response is None:
                # logger.debug("Start loading files because it doesn't exist in db")
                file_loader = S3FileLoader(
                    bucket=BUCKET_NAME,
                    key=obj['Key'],
                    endpoint_url=endpoint_s3,
                    aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
                    aws_secret_access_key=os.getenv("SCW_SECRET_KEY", "")
                )
                file_to_load = file_loader.load()
                cur.execute("INSERT INTO object_loaded (object_key) VALUES (%s)", (obj['Key'],))
                chunks = text_splitter.split_text(file_to_load[0].page_content)
                try:
                    embeddings_list = [embeddings.embed_query(chunk) for chunk in chunks]
                    vector_store.add_embeddings(chunks, embeddings_list)
                    cur.execute("INSERT INTO object_loaded (object_key) VALUES (%s)",
                                (obj['Key'],))
                except Exception as e:
                    print(e)
    conn.commit()
    llm = ChatOpenAI(
        base_url=os.getenv("SCW_INFERENCE_DEPLOYMENT_ENDPOINT"),
        api_key=os.getenv("SCW_SECRET_KEY"),
        model=deployment.model_name,
    )
    return deployment, vector_store, llm





