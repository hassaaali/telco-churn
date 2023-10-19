import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from definitions import root_dir
import os
from app.features import feats, target
from app.logger_engine import logger
import numpy as np
import pickle
import config
import joblib
import datetime
import tensorflow as tf


class ChurnRatePredictor:

    def __init__(self,
                 data_set: pd.DataFrame = None,
                 features: list = None,
                 target: str = None):
        self.log = logger.bind(Process='Churn Rate Predictor')
        self.dataset = data_set
        self.features = features
        self.target = target
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.random_forest_model = None
        self.neural_network_model = None
        self.confusion_matrix = None
        self.classification_report = None
        self.encoder_path = config.encoder_path
        self.random_forest_model_path = config.rf_model_path
        self.neural_network_model_path = config.nn_model_path
        self.label_encoders = {}

    def pre_process(self):
        self.log.info("Pre-Processing")
        all_fields = self.features + [target]
        for f in all_fields:
            le = LabelEncoder()
            le.fit(self.dataset[f])
            self.label_encoders[f] = le
            self.dataset[f] = le.transform(self.dataset[f])
            feat_encoder_file_path = os.path.join(self.encoder_path, "{n}_encoder.pkl".format(n=f))
            if f != self.target:
                with open(feat_encoder_file_path, "wb") as f:
                    joblib.dump(le, f)
                    self.log.info("Encoder Saved: ",
                                  Path=feat_encoder_file_path)
        self.X = self.dataset.drop(self.target, axis=1)
        self.y = self.dataset[self.target]

    def samples_split(self,
                      test_size: int = 0.2,
                      random_state: int = 42):
        self.log.info("Sample Splitting")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def scale(self):
        self.log.info("Scaling")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_random_forest(self,
                            n_estimators: int = 100,
                            criterion: str = 'gini',
                            max_depth: int = None,
                            min_samples_split: int = 2,
                            min_samples_leaf: int = 1,
                            min_weight_fraction_leaf: float = 0.0,
                            max_features: int = 'sqrt',
                            max_leaf_nodes: int = None,
                            min_impurity_decrease: float = 0.0,
                            bootstrap: bool = True,
                            oob_score: bool = False,
                            n_jobs: int = None,
                            random_state: int = None,
                            verbose: int = 0,
                            warm_start: bool = False,
                            class_weight: dict = None,
                            ccp_alpha: float = 0.0,
                            max_samples: int = None
                            ):
        self.log.info("Training Random Forest")
        self.random_forest_model = RandomForestClassifier(n_estimators=n_estimators,
                                                          criterion=criterion,
                                                          max_depth=max_depth,
                                                          min_samples_split=min_samples_split,
                                                          min_samples_leaf=min_samples_leaf,
                                                          min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                          max_features=max_features,
                                                          max_leaf_nodes=max_leaf_nodes,
                                                          min_impurity_decrease=min_impurity_decrease,
                                                          bootstrap=bootstrap,
                                                          oob_score=oob_score,
                                                          n_jobs=n_jobs,
                                                          verbose=verbose,
                                                          warm_start=warm_start,
                                                          class_weight=class_weight,
                                                          ccp_alpha=ccp_alpha,
                                                          max_samples=max_samples,
                                                          random_state=random_state)
        self.random_forest_model.fit(self.X_train, self.y_train)

    def evaluate_random_forest(self):
        y_pred = self.random_forest_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion = confusion_matrix(self.y_test, y_pred)
        self.classification_report = classification_report(self.y_test, y_pred)
        self.log.info('Predicted on original Test Dataset: ', Accuracy=accuracy)

    def save_random_forest(self):
        joblib.dump(self.random_forest_model, self.random_forest_model_path)
        self.log.info('Random Forest Model Saved: ', File=self.random_forest_model_path)

    def load_random_forest(self):
        self.random_forest_model = joblib.load(self.random_forest_model_path)

    def predict_random_forest(self,
                              new_data):
        all_files = os.listdir(self.encoder_path)
        features_labelized = [f.split("_encoder")[0] for f in all_files]
        new_data = pd.DataFrame(new_data).T
        for i in range(len(all_files)):
            with open(os.path.join(self.encoder_path, all_files[i]), "rb") as f:
                le = joblib.load(f)
                self.log.info("Encoder Loaded: ",
                              Feature=features_labelized[i])
                new_data[features_labelized[i]] = le.transform(new_data[features_labelized[i]])
        churn = self.random_forest_model.predict(new_data)
        churn = ["No" if c == 0 else "Yes" for c in churn]
        self.log.info("Prediction done on new dataset.")
        return churn

    def train_neural_network(self,
                             learning_rate: float = 0.01,
                             loss: str = "binary_crossentropy",
                             epochs: int = 100,
                             batch_size: int = 2):
        self.neural_network_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.neural_network_model.compile(optimizer=custom_optimizer,
                                          loss=loss,
                                          metrics=['accuracy'])
        self.log.info("Tensor Flow NN Training Beginning")
        self.neural_network_model.fit(self.X_train,
                                      self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(self.X_test, self.y_test))

    def evaluate_neural_network(self):
        loss, accuracy = self.neural_network_model.evaluate(self.X_test, self.y_test)
        self.log.info("Neural Network Accuracy: ", Accuracy=accuracy)

    def save_neural_network(self):
        self.neural_network_model.save(self.neural_network_model_path)
        self.log.info("Neural Network Model Saved: ", Path=self.neural_network_model_path)

    def load_neural_network_model(self):
        self.neural_network_model = tf.keras.models.load_model(self.neural_network_model_path)

    def predict_neural_network_model(self,
                              new_data):
        all_files = os.listdir(self.encoder_path)
        features_labelized = [f.split("_encoder")[0] for f in all_files]
        new_data = pd.DataFrame(new_data).T
        for i in range(len(all_files)):
            with open(os.path.join(self.encoder_path, all_files[i]), "rb") as f:
                le = joblib.load(f)
                self.log.info("Encoder Loaded: ",
                              Feature= features_labelized[i])
                new_data[features_labelized[i]] = le.transform(new_data[features_labelized[i]])
        new_data = new_data.to_numpy().astype('float32')
        churn = self.neural_network_model.predict(new_data)
        churn = ["No" if c == 0 else "Yes" for c in churn]
        self.log.info("Prediction done on new dataset.")
        return churn


if __name__ == "__main__":
    data_set = pd.read_csv(os.path.join(root_dir,
                                        "data",
                                        "data_telco_customer_churn.csv"),
                           low_memory=False)
    random_sample = data_set.iloc[0].iloc[:-1]
    c = ChurnRatePredictor(data_set=data_set,
                           features=feats,
                           target=target)
    c.pre_process()
    c.samples_split()
    c.scale()
    c.train_random_forest(n_estimators=100,
                          random_state=42,
                          criterion='gini',
                          min_samples_split=2)
    c.evaluate_random_forest()
    c.save_random_forest()
    c.load_random_forest()
    print(c.predict_random_forest(new_data=random_sample))
    c.train_neural_network(epochs=5, batch_size=1000)
    c.evaluate_neural_network()
    c.save_neural_network()
    c.load_neural_network_model()
    print(c.predict_neural_network_model(new_data=random_sample))
