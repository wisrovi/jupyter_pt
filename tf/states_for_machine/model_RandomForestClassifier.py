import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

from states_for_machine.utils import Utils

area_calculate = Utils.area_calculate
procrustes_calculate = Utils.procrustes_calculate

from typing import List, Tuple
import pickle, math

import warnings
warnings.filterwarnings('ignore')


class Model_RandomForestClassifier:
    max_points = 0
    family = []
    _normalization = []

    def __init__(self, clusters=None):
        self.clusters = clusters
    
    def prepare_train(self, trayectories:list) -> list:
        """
        Represent each trajectory with reference points

        @type trayectories: list
        @param trayectories: list of trajectories

        @rtype: list
        @returns: list of reference points
        """

        # find the maximum number of points in a trajectory
        max_points = max(len(trace) for trace in trayectories)
        
        # represent each trace with reference points
        # (fill with zeros if the trajectory is shorter than the maximum)
        reference_points = np.zeros((len(trayectories), max_points, 2))
        for i, trace in enumerate(trayectories):
            reference_points[i, :len(trace)] = trace
        
        # flatten the matrix to train the model
        data_to_train = reference_points.reshape(len(trayectories), -1)
        self.max_points = max_points
        return data_to_train
    
    def prepare_predict(self, news_trayectories:list) -> list:
        """
        Represent each trajectory with reference points

        @type news_trayectories: list
        @param news_trayectories: list of trajectories

        @rtype: list
        @returns: list of reference points
        """
        
        # represent each trace with reference points
        reference_points_nuevas = np.zeros((len(news_trayectories), self.max_points, 2))
        for i, trace in enumerate(news_trayectories):
            reference_points_nuevas[i, :len(trace)] = trace
        
        # flatten the matrix to train the model
        datos_prediccion = reference_points_nuevas.reshape(len(news_trayectories), -1)
        return datos_prediccion    

    def create_model(self, X_train:list, y_train:list) -> "RandomForestClassifier":
        """
        Create a random forest model

        if is necessary, you can change the algorithm for another one
        for example logistic regression, svm, etc.
        
        @type X_train: list
        @param X_train: list of reference points

        @type y_train: list
        @param y_train: list of labels

        @rtype: RandomForestClassifier
        @returns: random forest model
        """

        # Create the model with 100 trees and a random seed of 42
        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # train the model with training data
        modelo_rf.fit(X_train, y_train)
        return modelo_rf

    def train(self, X, y, test_all=False, verbose=False):
        """
        Train the model

        @type X: list
        @param X: list of reference points

        @type y: list
        @param y: list of labels

        @type test_all: bool
        @param test_all: if True, the model will be tested with all the data

        @type verbose: bool
        @param verbose: if True, the model will show the results of the training

        @rtype: tuple
        @returns: tuple with the classification report and the confusion matrix
        """

        self.X = X
        self.y = y
        X = self.prepare_train(X)
        
        # split the data into training and test sets
        if verbose:
            print("Before split:", Counter(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     
        
        if verbose:
            print("After split:", Counter(y_train), " - ", Counter(y_test))

        modelo_rf = self.create_model(X_train, y_train)

        if test_all:
            y_pred = modelo_rf.predict(X)
            y_test = y
        else:
            y_pred = modelo_rf.predict(X_test)
        
        # Calculate the precision of the model
        precision = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {precision:.2f}")
        
        # show the classification report
        report = classification_report(y_test, y_pred)
        m_c = confusion_matrix(y_test, y_pred)

        self.modelo_rf = modelo_rf
        
        self.family = list(set(y))
        
        return report, m_c

    def predict(self, data_predict:list) -> list:
        """
        Predict the labels of the new data

        @type data_predict: list
        @param data_predict: list of new trajectories

        @rtype: list
        @returns: list of labels
        """
        prediction_data = self.prepare_predict(data_predict)

        # predict the labels of the new data
        y_pred = self.modelo_rf.predict(prediction_data)
        return y_pred

    @property
    def get_family(self) -> list:
        """
        Get the family of labels

        @rtype: list
        @returns: list of labels
        """
        return self.family

    def get_brothers(self, cluster:int) -> list:
        """
        Get the trajectories of especified cluster

        @type cluster: int
        @param cluster: cluster number

        @rtype: list
        @returns: list of trajectories
        """
        brothers = []
        for i in self.clusters[cluster]:
            brothers.append( self.X[i] )
        return brothers

    @property
    def normalization(self) -> list:
        """
        Get the normalization of the data

        @rtype: list
        @returns: list of normalization
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value:list):
        """
        Set the normalization of the data

        @type value: list
        @param value: list of normalization
        """
        self._normalization = value

    def save(self, path:str = None):
        """
        Save the model

        @type path: str
        @param path: path to save the model
        """
        if not path:
            path = "Model_RandomForestClassifier.pkl"

        with open(path, 'wb') as file:
            pickle.dump([
                self.modelo_rf, 
                self.X,
                self.y,
                self.max_points,
                self.clusters,
                self._normalization
            ], file)

    def load(self, path:str = None):
        """
        Load the model of especified path

        @type path: str
        @param path: path to load the model
        """
        if not path:
            path = "Model_RandomForestClassifier.pkl"

        with open(path, 'rb') as file:
            self.modelo_rf, self.X, self.y, self.max_points, self.clusters, self._normalization = pickle.load(file)

    def evaluate_anomaly(self, data_predict:List[Tuple[int, int]]) -> Tuple[bool, bool]:
        """
        Evaluate the anomaly of the new data

        @type data_predict: list
        @param data_predict: list of new trajectories

        @rtype: tuple
        @returns: tuple with the social trayectory anomaly and the social area anomaly
        """
        if len(self.normalization) == 0:
            raise Exception("No normalization")

        y_pred = self.predict([data_predict])[0]
        brothers = self.get_brothers(y_pred)
        normalization = self.normalization[y_pred]

        pred_procrustes = []
        pred_areas = []
        for line2 in brothers:
            area = area_calculate(  data_predict, line2, normalization[4] )
            if not math.isnan(  area[0]   ):          
                try:
                    procruste = procrustes_calculate(  data_predict, line2, normalization[1] )    
        
                    pred_procrustes.append(procruste[0])
                    pred_areas.append(area[0])
                except:
                    pass
            
        result = (
            round(
                sum(pred_procrustes)/len(pred_procrustes), 
                3
            ), 
            round(
                sum(pred_areas)/len(pred_areas), 
                3
            )
        )
        
        social_trayectory_anomaly = result[0] > normalization[1]
        social_area_anomaly = result[1] > normalization[4]

        return social_trayectory_anomaly, social_area_anomaly

