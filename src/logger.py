import logging
import os
from datetime import datetime

Log_File=f"{datetime.now().strftime(('%m_%d_%Y_%H_%M_%S'))}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,Log_File)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
    )