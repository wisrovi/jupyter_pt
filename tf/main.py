from states_for_machine.extend_data import ExtendData
from states_for_machine.extra_features import ExtraFeatures
from states_for_machine.individual_anomaly import IndividualAnomaly

from state_machine.Machine import Machine

import pickle
import pandas as pd


def get_info_vehicle():
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


def get_shape_image():
    return (1080, 1920)


class Brain:
    path_calibrations = "/serverstorage/dataset_tailandia/calibration_dataset.pkl"

    @property
    def machine(self):
        return Machine(state=[
                ExtendData(next_state="ExtraFeatures"), 
                ExtraFeatures(next_state="IndividualAnomaly"),
                IndividualAnomaly(next_state=None)
            ], initial='ExtendData')

    @property
    def calibrations(self):
        with open(self.path_calibrations, 'rb') as archivo:
            return pickle.load(archivo)


brain = Brain()
brain.path_calibrations = "/serverstorage/dataset_tailandia/calibration_dataset.pkl"

data = {
    "data": get_info_vehicle(),
    "size_frame": get_shape_image(),  # visual reference or observation point,
    "config": {
        "threshold_similarity": 45,  # Adjust the threshold value according to your needs, the higher the value the fewer groups it creates
        "umbral_anomaly": 1
    },    
    "calibrations": brain.calibrations
}


machine = brain.machine
result = machine.cicle(**data)


del result["calibrations"] 

print(" \n"*5)
print(result.keys())
print(result["anomaly"])


import json
print(result)



import cv2
import matplotlib.pyplot as plt
fondo = "/serverstorage/Tailandia/Tailandia 001.jpg"
fondo = cv2.imread(fondo)
plt.figure(figsize=(20,8))
plt.title(f'Movimientos actuales del vehiculo')
plt.imshow(fondo)
plt.savefig(f"/serverstorage/{DATASET}_{type_vehicle_graph}.png")
#plt.show()