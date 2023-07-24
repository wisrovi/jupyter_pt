from state_machine.State import State
import pickle



class Capture_data(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        this method reads the dataset of the selected sources, 
        extracts the coordinates of the points of each line and saves them in a pickle file, 
        it includes a dictionary with the names of the lines and the lines

        you can to change the origen of data according to your needs, 
        but the output must be a dictionary with the following structure:
        {
            "all_lines": [
                [
                    [x1, y1],
                    [x2, y2],
                    ...
                ],
                ...
            ],
            "all_names": [
                "name1",
                "name2",
                ...
            ]
        }
        
        @type kwargs: dict
        @param kwargs: dict with data to process

        @rtype: dict
        @returns: dict with all_lines and all_names for train social cluster

        """  
        
        all_lines_saved, all_names_saved = None, None

        with open("/serverstorage/dataset_tailandia/calibration_dataset.pkl", 'rb') as archivo:
            all_lines_saved, all_names_saved = pickle.load(archivo)

        # for this dataset, there are some anomalies that must be removed for the model training to work correctly
        anomalies = {
            "0082": None,
            "0130": None,
            "0317": None,
            "0326": None,
            "0275": None,
        }

        for name_delete in anomalies.keys():
            if name_delete in all_names_saved:
                index = all_names_saved.index(name_delete)
                if index >= 0:
                    all_names_saved.remove(name_delete)
                    anomalies[name_delete] = all_lines_saved.pop(index)
                    # print(f"remove {name_delete}")

        # when the dataset is cleaned, it is saved again in variable according to the structure of the output
        kwargs["dataset"] = {
            "all_lines": all_lines_saved,
            "all_names": all_names_saved
        }

        return kwargs