from state_machine.State import State

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN

class IndividualAnomaly(State):
    
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
        
        if not kwargs.get("features"):
            raise Exception("No features in kwargs")

        kwargs["anomaly"] = {
            "individual": {}
        }

        #X = kwargs.get("features").get("array")
        X = kwargs.get("features").get("features_np").tolist()

        try:
            modelo = IsolationForest(contamination=0.1) 
            modelo.fit(X)
            y_pred_IsolationForest = modelo.predict(X).tolist()
        except:
            y_pred_IsolationForest = []
    
        try:
            modelo = EllipticEnvelope(contamination=0.1)  # Define la proporción de datos considerados anómalos
            modelo.fit(X)
            y_pred_EllipticEnvelope = modelo.predict(X).tolist()
        except:
            y_pred_EllipticEnvelope = []
    
        try:
            modelo = DBSCAN(eps=0.5, min_samples=2) 
            modelo.fit(X)
            y_pred_DBSCAN = modelo.labels_.tolist()
        except:
            y_pred_DBSCAN = []
    
        votes = []
        if len(y_pred_IsolationForest) > 0:
            if len(y_pred_IsolationForest) == len(y_pred_EllipticEnvelope) == len(y_pred_DBSCAN):
                for i in range(len(y_pred_DBSCAN)):                
                    count = 0
                    for tmp_list in [y_pred_IsolationForest, y_pred_EllipticEnvelope, y_pred_DBSCAN]:
                        if tmp_list[i] == -1:
                            count += 1
                    votes.append(count)

        kwargs["anomaly"]["individual"]["IsolationForest"] = y_pred_IsolationForest
        kwargs["anomaly"]["individual"]["EllipticEnvelope"] = y_pred_EllipticEnvelope
        kwargs["anomaly"]["individual"]["DBSCAN"] = y_pred_DBSCAN
        kwargs["anomaly"]["individual"]["summary"] = votes
        kwargs["anomaly"]["individual"]["there_are"] = len(list(set(votes))) > 0

        return kwargs