import os
import sys
sys.path.append('C:\\Users\\Mayan\\OneDrive\\Documents\\Machine Learning\\End2End Projects')  # Adjust the path accordingly
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation, dataclass, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

#from exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        # Setting Up Configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the dataset
            df = pd.read_csv('Notebook\\data\\exams.csv')
            logging.info('Read the dataset as dataframe')

            # Creating directories if not present
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data ingestion completed")

            # Train-test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    #Initiating data ingestion function
    train_data,test_data=obj.initiate_data_ingestion()
    logging.info("Data Ingestion completed successfully")


    #parameters for data transformation
    train_path = 'artifacts\\train.csv'
    test_path = 'artifacts\\test.csv'
    numerical_columns = ["writing_score", "reading_score"]
    categorical_columns = ["gender", "ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
    target_column_name = ['math_score']

    logging.info("Parameters initialized successfully")


    data_transformation=DataTransformation()
    train_arr,test_arr,preprocessor_file_path=data_transformation.initiate_data_transformation(
        train_path, 
        test_path, 
        target_column_name, 
        numerical_columns, 
        categorical_columns
        )

    logging.info('Data Transformation is complete')

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))