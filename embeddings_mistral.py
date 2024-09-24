import logging
import os


import psycopg2
from langchain_postgres import PGVector
from scaleway import Client
from scaleway.inference.v1beta1.api import InferenceV1Beta1API
from scaleway.inference.v1beta1.types import EndpointSpec
from scaleway.inference.v1beta1.types import EndpointSpecPublic
from scaleway.rdb.v1.api import RdbV1API
from scaleway.rdb.v1.types import Permission
from dotenv import load_dotenv
from langchain_community.document_loaders import S3DirectoryLoader, S3FileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

MODEL_NAME = "mistral/mistral-7b-instruct-v0.3:bf16"
NODE_TYPE = "L4"
BUCKET_NAME = "rag-test"
DEPLOYMENT_NAME = "RAG"
INSTANCE_NAME = "RAG-instance"
DB_NAME = "DB_RAG"
DB_USER = "laure-di"
MODEL_NAME_EMBED="sentence-transformers/sentence-t5-xxl:fp32"
DEPLOYMENT_NAME_EMBED= "RAG-embeddings"

logger = logging.getLogger('scaleway')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

load_dotenv()

def deployment_exists(deployment_name):
    deployments = inference_api.list_deployments()
    if deployments.total_count == 0:
        return False
    for deployment_inference in deployments.deployments:
        if deployment_inference.name == deployment_name:
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

def instance_by_name():
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
    exist = deployment_exists(DEPLOYMENT_NAME)
    if not exist:
        deployment = inference_api.create_deployment(model_name=MODEL_NAME, node_type=NODE_TYPE, endpoints=[endpoint],
                                                 project_id=project_id, name=DEPLOYMENT_NAME)
    else:
        deployment = deployment_by_name(DEPLOYMENT_NAME)

    exist = deployment_exists(DEPLOYMENT_NAME_EMBED)
    if not exist:
        deployment_embeddings = inference_api.create_deployment(model_name=MODEL_NAME_EMBED, node_type=NODE_TYPE, endpoints=[endpoint],
                                                     project_id=project_id, name=DEPLOYMENT_NAME_EMBED)
    else:
        deployment_embeddings = deployment_by_name(DEPLOYMENT_NAME_EMBED)

    logger.debug("Create or get DB")
    instance_exist = instance_exists()
    if not instance_exist:
        instance_db = db_api.create_instance(engine="PostgreSQL-15", user_name="laure-di", password=os.getenv("SCW_DB_PASSWORD"), node_type="db-play2-nano", region="fr-par", is_ha_cluster=False,disable_backup=True, volume_size=10000000000, volume_type="bssd", backup_same_region=False, name=INSTANCE_NAME)
        instance_db = db_api.wait_for_instance(instance_id=instance_db.id)
        db = db_api.create_database(instance_id=instance_db.id, name=DB_NAME)
        db_api.set_privilege(instance_id=instance_db.id, database_name=DB_NAME, user_name="laure-di", permission=Permission.ALL)
    else:
        instance_db = instance_by_name()
        db = db_by_instance(instance_db.id)


    logger.debug("Start trying to connect to database")
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=os.getenv("SCW_DB_PASSWORD"),
        host=instance_db.endpoint.ip,  # Use the public/private IP or hostname
        port=instance_db.endpoint.port  # Use the correct port
    )
    cur = conn.cursor()
    logger.debug("Connected to database")

    logger.debug("Install pg_vector extension")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


    logger.debug("Create table object_key")
    cur.execute("""CREATE TABLE IF NOT EXISTS object_loaded (id SERIAL PRIMARY KEY, object_key TEXT)""")


    endpoint_s3 = "https://s3." + os.getenv("SCW_DEFAULT_REGION", "") + ".scw.cloud"
    documentLoader = S3DirectoryLoader(bucket=BUCKET_NAME, endpoint_url=endpoint_s3,
                                       aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
                                       aws_secret_access_key=os.getenv("SCW_SECRET_KEY", ""))


    embedding_url = deployment_embeddings.endpoints[0].url + "/v1"
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("SCW_API_KEY_EMBED"),
        openai_api_base=embedding_url,
        model="sentence-transformers/sentence-t5-xxl",
        tiktoken_enabled=False,
    )

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    connection_string = "postgresql+psycopg2://" + conn.info.user + ":" + conn.info.password + "@" + conn.info.host + ":" + str(
        conn.info.port) + "/" + conn.info.dbname
    vector_store = PGVector(
        connection=connection_string,
        embeddings=embeddings,
    )

    logger.debug("start loading documents lazy way")
    files = documentLoader.lazy_load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    for file in files:
        cur.execute("SELECT object_key FROM object_loaded WHERE object_key = %s", (file.metadata["source"],))
        if cur.fetchone() is None:
            logger.debug("start loading files because it doesn't exist in db")

            key_file = file.metadata["source"].split("/")[-1]
            fileLoader = S3FileLoader(bucket=BUCKET_NAME, key=key_file ,endpoint_url=endpoint_s3,
                                      aws_access_key_id=os.getenv("SCW_ACCESS_KEY", ""),
                                      aws_secret_access_key=os.getenv("SCW_SECRET_KEY", ""))
            file_to_load = fileLoader.load()
            print(file_to_load)
            cur.execute("INSERT INTO object_loaded (object_key) VALUES (%s)", (file.metadata["source"],))
            chunks = text_splitter.split_text(file.page_content)
            try:
                logger.debug("start embedding chunks")
                embeddings_list = [embeddings.embed_query(chunk) for chunk in chunks]
                for embeddings, chunk in embeddings_list, chunks:
                    logger.debug("add embeddings to db")
                    vector_store.add_embeddings(embeddings, chunk)
            except Exception as e:
                logging.error(f"An error occurred: {e}")

    conn.commit()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    deployment_url = deployment.endpoints[0].url + "/v1"
    llm = ChatOpenAI(
        base_url=deployment_url,
        api_key=os.getenv("SCW_API_KEY"),
        model=deployment.model_name,
        temperature=0.0,
    )

    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
    )

    query = "What are the step to setup a database with the cli of scaleway?"

    response = qa_stuff.invoke(query)

    print(response['result'])