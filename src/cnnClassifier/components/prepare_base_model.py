from pathlib import Path
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from keras import backend as K
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_source_path)
        ft_input = self.model.input
        ft_layers = self.model.layers
        self.model = tf.keras.models.Model(inputs = ft_input, outputs = ft_layers[-4].output)
        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, learning_rate, freeze_till = None):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        feature_in = tf.keras.layers.Dropout(0.3)(model.output)
        feature_in = tf.keras.layers.BatchNormalization()(feature_in)
        prediction = tf.keras.layers.Dense(
            units=1,
            activation="sigmoid"
        )(feature_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["binary_accuracy", f1_m]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
