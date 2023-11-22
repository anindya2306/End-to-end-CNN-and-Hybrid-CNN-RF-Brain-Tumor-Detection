from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import pickle



class PredictionPipeline(PrepareBaseModel):
    def __init__(self,filename):
        self.filename =filename
        self.__classifier_path = "lgbm_classifier.pkl"
        self.__feature_dim = 64


    
    def predict(self):
        # load model
        feature_model = super().get_base_model()

        with open(self.__classifier_path, "wb") as f:
            lgbm_model = pickle.load(f)

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (240,240))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        feat = feature_model.predict(test_image).reshape(-1, self.__feature_dim)
        result = lgbm_model.predict(feat)

        print(result[0])

        if result[0] == 0:
            prediction = 'Healthy'
            return [{ "image" : prediction}]
        else:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
