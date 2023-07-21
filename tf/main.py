from states_for_machine.extend_data import ExtendData
from states_for_machine.extra_features import ExtraFeatures
from states_for_machine.individual_anomaly import IndividualAnomaly
from states_for_machine.predict_cluster_social import Predict_cluster_social

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
                IndividualAnomaly(next_state="Predict_cluster_social"),
                Predict_cluster_social(next_state=None),
            ], initial='ExtendData')

    @property
    def calibrations(self):
        with open(self.path_calibrations, 'rb') as archivo:
            all_lines_saved, all_names_saved = pickle.load(archivo)
            return {
                "all_lines": all_lines_saved,
                "all_names": all_names_saved
            }


brain = Brain()
brain.path_calibrations = "/serverstorage/dataset_tailandia/calibration_dataset.pkl"

data = {
    "data": get_info_vehicle(),
    "size_frame": get_shape_image(),  # visual reference or observation point,
    "config": {
        "social_sensibility": 50,  # Adjust the threshold value according to your needs, the higher the value the fewer clusters it creates
        "umbral_individual_anomaly": 1  # Adjust the umbral_individual_anomaly value according to your individual sensibility, recommended value 1
    },    
    "calibrations": brain.calibrations  # lines for define the normal social cluster, this lines are coordinates of the center of each vehicle in normal conduction
}


machine = brain.machine
result = machine.cicle(**data)


del result["calibrations"] 
del result["config"] 
del result["time"] 
del result["size_frame"] 

print(" \n"*5)
print(result.keys())
print(result["anomaly"])


exit()

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