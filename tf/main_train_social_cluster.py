from states_for_machine.capture_data import Capture_data
from states_for_machine.cluster_with_algoritmia import Cluster_with_algortimia
from states_for_machine.transfer_learning_to_model import Transfer_learning_to_model
from states_for_machine.transfer_learning_to_model import Transfer_learning_to_model
from states_for_machine.extract_normality import Extract_normality



from state_machine.Machine import Machine


class Brain:

    @property
    def machine(self):
        return Machine(state=[
                Capture_data(next_state="Cluster_with_algortimia"),
                Cluster_with_algortimia(next_state="Transfer_learning_to_model"),
                Transfer_learning_to_model(next_state="Extract_normality"),
                Extract_normality(next_state=None),
            ], initial='Capture_data')


data = {
    "sensitive": 50.0, # Adjust the threshold value according to your needs, 
                    # the higher the value the fewer clusters it creates
    "path_save": "Model_RandomForestClassifier.pkl",
}

brain = Brain()
result = brain.machine.cicle(**data)
# print(result.keys())

if result["tl_model"]["status_save"]:
    print("model saved")
