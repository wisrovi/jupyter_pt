from states_for_machine.extend_data import ExtendData
from states_for_machine.extra_features import ExtraFeatures
from states_for_machine.individual_anomaly import IndividualAnomaly
from states_for_machine.social_anomaly_predict import Social_anomaly

from state_machine.Machine import Machine

import pickle
import pandas as pd


def get_info_vehicle() -> list:
    """
    method to get info vehicle, you can replace this method for your method to get info vehicle, for example,
    you can use a method to get info vehicle from a video or from a camera with tracking algorithm

    the structure for return is:
    [
        {
            "center": (x, y),
            "time": time
        },
        {
            "center": (x, y),
            "time": time
        },
        ...
    ]

    @rtype: list
    @returns: list with info vehicle
    """
    dataframe = pd.read_csv("/serverstorage/dataset_tailandia/0246.csv")

    data_to_process = []
    for index, row in dataframe.iterrows():
        center = (row["original_coordinate_x"], row["original_coordinate_y"])
        time = row["time"]
        data_to_process.append({
            "center": center,
            "time": time
        })
    return data_to_process


def get_visual_observation_point():
    return (540, 960)


class Brain:
    @property
    def machine(self):
        return Machine(state=[
                ExtendData(next_state="ExtraFeatures"), 
                ExtraFeatures(next_state="IndividualAnomaly"),
                IndividualAnomaly(next_state="Social_anomaly"),
                Social_anomaly(next_state=None),
            ], initial='ExtendData')


brain = Brain()

data = {
    "data": get_info_vehicle(),  # Data to process to find anomaly according to conduction behavior
    "visual_observation_point": get_visual_observation_point(),  # visual reference or observation point,
    "config": {
        "path_saved_model": "Model_RandomForestClassifier.pkl",  # Path to the model to be used for social anomaly detection
        "umbral_individual_anomaly": 1  # Adjust the umbral_individual_anomaly value according to your individual sensibility, recommended value 1
    },        
}


machine = brain.machine
result = machine.cicle(**data)


del result["config"] 
del result["time"] 
del result["visual_observation_point"] 

print(" \n"*5)
print(result.keys())
print(result["data"])
print(result["anomaly"])
