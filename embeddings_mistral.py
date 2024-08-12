import configparser
import logging
import os
from scaleway import Client
from scaleway.inference.v1beta1.types import CreateDeploymentRequest

logger = logging.getLogger('scaleway')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class RAGTest:
    client: Client
    Deployment: CreateDeploymentRequest

    def __init__(self, config_file=None, model_name: str = None, node_type: str = None):
        self.client = None
        self.Deployment = None
        if config_file and os.path.exists(config_file):
            logger.debug(f"Creating client from file: {config_file}")
            self._create_client_from_file(config_file)
        else:
            logger.debug("Creating client from environment variables")
            self._create_client_from_env()
        if model_name is None or node_type is None:
            raise ValueError('model_name or node_type must be specified')

    def _create_client_from_file(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.client = Client(
            access_key=config.get('DEFAULT', 'ACCESS_TOKEN', fallback=""),
            secret_key=config.get('DEFAULT', 'SECRET_KEY', fallback=""),
            default_project_id=config.get('DEFAULT', 'PROJECT_ID', fallback=""),
            default_region=config.get('DEFAULT', 'DEFAULT_REGION', fallback=""),
            default_zone=config.get('DEFAULT', 'DEFAULT_ZONE', fallback=""),
        )
        self.client.validate_credentials()
        logger.debug(f"Client created with access key: {self.client.access_key}")

    def _create_client_from_env(self):
        try:
            self.client = Client(
                access_key=os.getenv("ACCESS_TOKEN"),
                secret_key=os.getenv("SECRET_KEY"),
                default_project_id=os.getenv("PROJECT_ID"),
                default_region=os.getenv("DEFAULT_REGION"),
                default_zone=os.getenv("DEFAULT_ZONE"),
            )
            self.client.validate_credentials()
        except ValueError as err:
            raise ValueError(f"Error initializing client with provided config: {err}")


try:
    rag = RAGTest()
    print("access key:", rag.client.access_key)  # Attempt to access client's access_key
except ValueError as e:
    print(f"Error: {e}")
except RuntimeError as e:
    print(f"Initialization Error: {e}")
except AttributeError as e:
    print(f"Attribute Error: {e}")
