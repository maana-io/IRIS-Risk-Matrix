import os
from dotenv import load_dotenv
import logging


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
load_dotenv(verbose=True, dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

SERVICE_ID = os.getenv('SERVICE_ID')
SERVICE_ADDRESS = os.getenv('SERVICE_ADDRESS')
SERVICE_PORT = os.getenv('SERVICE_PORT')

LOG_LEVEL = logging.DEBUG

LOW_CONFIDENCE_THRESHOLD = float(os.getenv('LOW_CONFIDENCE_THRESHOLD'))

IRIS_UNIQUE_ID = os.getenv('IRIS_UNIQUE_ID')
IRIS_SEVERITY_LIKELIHOOD_MODEL_ID = os.getenv('IRIS_SEVERITY_LIKELIHOOD_MODEL_ID')
