from state_machine.State import State
from states_for_machine.cluster_algoritmia import ClusterAlgoritmia


import warnings
warnings.filterwarnings('ignore')


class Predict_cluster_social(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method to execute state for locate vehicles in frame

        @type kwargs: dict
        @param kwargs: dict with data to process
        @param example: a example for input data in file "example_input_kwargs_car_detection.json"

        @rtype: dict
        @returns: dict with data processed, add vehicles in frame if exist
        @return example: a example for output data in file "example_output_kwargs_car_detection.json"
        """  
        
        if not kwargs.get("data"):
            raise Exception("No data in kwargs")

        if not kwargs.get("config"):
            raise Exception("No config in kwargs")

        if not kwargs.get("config").get("social_sensibility"):
            raise Exception("No social_sensibility in config in kwargs")

        data_predict = [ id["center"] for id in kwargs.get("data") ] 

        sensibility = kwargs.get("config").get("social_sensibility")
        all_lines_calibrations = kwargs.get("calibrations").get("all_lines")
        all_names_calibrations = kwargs.get("calibrations").get("all_names")

        k_william = ClusterAlgoritmia(sensibility=sensibility)
        k_william.train(all_lines_calibrations)

        cluster = k_william.predict([data_predict])[0]
        brothers = k_william.brothers(cluster)

        kwargs["anomaly"]["social"] = {
            "cluster": cluster,
            "brothers": brothers
        }

        return kwargs