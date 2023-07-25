from state_machine.State import State
from states_for_machine.model_RandomForestClassifier import Model_RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')


class Social_anomaly(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method that loads a trained model to detect the cluster according to the driving behavior, 
        then compares the behavior with its cluster and determine if there is a social anomaly

        @type kwargs: dict
        @param kwargs: dict with data to process

        @rtype: dict
        @returns: dict with data processed, add vehicles in frame if exist
        """  
        
        if not kwargs.get("data"):
            raise Exception("No data in kwargs")

        if not kwargs.get("config"):
            raise Exception("No config in kwargs")

        if not kwargs.get("config").get("path_saved_model"):
            raise Exception("No path_saved_model in config in kwargs")

        data_predict = [ id["center"] for id in kwargs.get("data") ] 

        model = Model_RandomForestClassifier()
        model.load(kwargs.get("config").get("path_saved_model"))

        social_trayectory_anomaly, social_area_anomaly = model.evaluate_anomaly(data_predict)

        kwargs["anomaly"]["social"] = {
            "trayectory": social_trayectory_anomaly,
            "area": social_area_anomaly
        }

        return kwargs