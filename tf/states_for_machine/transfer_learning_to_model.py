from states_for_machine.model_RandomForestClassifier import Model_RandomForestClassifier


from state_machine.State import State


class Transfer_learning_to_model(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method to prepare data for train social cluster

        @type kwargs: dict
        @param kwargs: dict with data to process

        @rtype: dict
        @returns: dict with all_lines and all_names for train social cluster
        """  
        
        if not kwargs.get("dataset"):
            raise Exception("No dataset in kwargs")

        if not kwargs.get("dataset").get("clusters"):
            raise Exception("No dataset.clusters in kwargs")

        if not kwargs.get("dataset").get("X"):
            raise Exception("No dataset.X in kwargs")

        if not kwargs.get("dataset").get("y"):
            raise Exception("No dataset.y in kwargs")


        clusters = kwargs.get("dataset").get("clusters")
        X = kwargs.get("dataset").get("X")
        y = kwargs.get("dataset").get("y")


        model = Model_RandomForestClassifier(clusters)
        classification_report, confusion_matrix = model.train(X, y, test_all=True)

        if kwargs.get("verbose"):
            print("Reporte de clasificación:")
            print(classification_report)

        kwargs["tl_model"] = {
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "model": model
        }

        return kwargs