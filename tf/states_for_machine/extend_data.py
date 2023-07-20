from state_machine.State import State
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')


class ExtendData(State):
    
    cantidad_predicciones=2
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method to execute state for extend data using ARIMA (temporal series)

        @type kwargs: dict
        @param kwargs: dict with data to process
        @param example: a example for input data in file "example_input_kwargs_car_detection.json"

        @rtype: dict
        @returns: dict with data processed, add vehicles in frame if exist
        @return example: a example for output data in file "example_output_kwargs_car_detection.json"
        """  
        (p, d, q) = (1, 1, 2)
        
        if not kwargs.get("data"):
            raise Exception("No data in kwargs")

        original_data = [ id["center"] for id in kwargs.get("data") ] 
        times = [t["time"] for t in kwargs["data"]][::-1]
        kwargs["time"] = [int(t) for t in times]
        #print(f"[ExtendData]: times: {times}")

        kwargs["coordinates"] = {
            "origin": original_data,
            "futures": {}
        }

        
        if len(original_data) > 2:
            model_time = ARIMA(times, order=(p, d, q))
            model_time = model_time.fit()
            model_time = model_time.get_forecast(steps=self.cantidad_predicciones)
            model_time = model_time.predicted_mean
            model_time = [ int(a) for a in  list(model_time)]
            model_time = [1 if x == 0 else x for x in model_time]
            model_time = model_time[::-1] + times[::-1]
            # print(f"[ExtendData]: model_time: {model_time}")
            kwargs["coordinates"]["futures"]["times"] = model_time
            
            all_x = []
            all_y = []
            for x, y in original_data:
                all_x.append(x)
                all_y.append(y)  

            for estado in ["after", "before"]:
                if estado == "before":
                    all_x = all_x.copy()[::-1]
                    all_y = all_y.copy()[::-1]

                model_x = ARIMA(all_x, order=(p, d, q))
                model_y = ARIMA(all_y, order=(p, d, q))
                
                model_x_fit = model_x.fit()
                model_y_fit = model_y.fit()
                
                forecast_x = model_x_fit.get_forecast(steps=self.cantidad_predicciones)
                forecast_y = model_y_fit.get_forecast(steps=self.cantidad_predicciones)
        
                predicted_values_x = forecast_x.predicted_mean
                predicted_values_y = forecast_y.predicted_mean
        
                predicted_values_x = [ int(a) for a in  list(predicted_values_x)]
                predicted_values_y = [ int(a) for a in  list(predicted_values_y)]
                
                predictions = list(zip(list(predicted_values_x), list(predicted_values_y)))
        
                if estado == "before":
                    predictions = predictions[::-1]

                kwargs["coordinates"]["futures"][estado] = predictions
                
            data = kwargs["coordinates"]["futures"]["before"] + kwargs["coordinates"]["origin"]+ kwargs["coordinates"]["futures"]["after"]       
            kwargs["coordinates"]["futures"]["p_p_f"] = data
        
        return kwargs