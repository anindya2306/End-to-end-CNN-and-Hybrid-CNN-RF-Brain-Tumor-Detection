import os
import urllib.request as request
import zipfile
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig, from_local_dataset =True):
        self.config = config
        self.from_local_dataset = from_local_dataset


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file) and not self.from_local_dataset:
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        elif not os.path.exists(self.config.local_data_file) and self.from_local_dataset:
            logger.info(f"Loading from local dataset of size: {get_size(Path(self.config.local_source_path))}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        if not self.from_local_dataset:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        else:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_source_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)


