from state_machine.State import State
import pandas as pd
import math

import warnings
warnings.filterwarnings('ignore')


class ExtraFeatures(State):

    TIME_FRAME = 1 # second
    
    def execute(self, **kwargs: dict) -> dict:
        """
        This method receives a series of movement coordinates defining the change 
        of time between each point and a reference point (usually the center of the frame) 
        and with these data, it is calculated:

        Distance from the current point and the previous point, both Pythagoric distance 
        and distance in X and Y, also the same 3 distances but with the reference point.

        speed of coordinate change, both Pythagorica and X and Y, both for the previous 
        point and with the reference point.

        The acceleration of coordinate change, both for the previous point and for the reference, 
        which shows both the change in pythagoric acceleration, in X and Y, in Y,,

        The point of the point with respect to the reference point

        the angles of movement of the point with respect to the previous point and 
        with respect to the reference

        @type kwargs: dict
        @param kwargs: dict with data to process
        @param example: a example for input data in file "example_input_kwargs_car_detection.json"

        @rtype: dict
        @returns: dict with data processed, add vehicles in frame if exist
        @return example: a example for output data in file "example_output_kwargs_car_detection.json"
        """  
        
        if not kwargs.get("coordinates"):
            raise Exception("No coordinates in kwargs")      

        if not kwargs.get("size_frame"):
            raise Exception("No size_frame in kwargs")

        if not kwargs["coordinates"]["futures"]["times"]:
            raise Exception("No times in kwargs")

        
        times = kwargs["coordinates"]["futures"]["times"]
        times = [times[i+1]-times[i] for i in range(len(times)-1)]
        times = [1 if x == 0 else x for x in times]
        times.insert(0, int(sum(times)/len(times)))
        times.insert(-1, int(sum(times)/len(times)))
        
        df = pd.DataFrame()
        
        array = kwargs["coordinates"]['origin'][:]
        ref = kwargs["size_frame"]
        
        original_size = len(kwargs["coordinates"]['origin'])
        
        quadrants = [ self.get_quadrant(p, ref) for p in array ]
        df["quadrants"] = quadrants
        df["time"] = [t["time"] for t in kwargs["data"]]
        
        df["original_coordinate_x"] = [p[0] for p in array]
        df["original_coordinate_y"] = [p[1] for p in array]

        zero0, zero1 = kwargs["coordinates"]["futures"]["before"]
        array.insert(0, zero1)
        array.insert(0, zero0)
        
        if True:            
            distances = [ self.distance(array[i], array[i+1]) for i in range(len(array) - 1)  ]
            x_distances = [ self.distanceX(array[i], array[i+1]) for i in range(len(array) - 1)  ]
            y_distances = [ self.distanceY(array[i], array[i+1]) for i in range(len(array) - 1)  ] 
            
            distances_ref = [ self.distance(array[i], ref) for i in range(len(array))  ]
            x_distances_ref = [ self.distanceX(array[i], ref) for i in range(len(array)) ]
            y_distances_ref = [ self.distanceY(array[i], ref) for i in range(len(array))  ]  
                        
            df["distances"] = distances[-original_size:]
            df["x_distances"] = x_distances[-original_size:]
            df["y_distances"] = y_distances[-original_size:]
            
            df["distances_ref"] = distances_ref[-original_size:]
            df["x_distances_ref"] = x_distances_ref[-original_size:]
            df["y_distances_ref"] = y_distances_ref[-original_size:]          
        
        if True:            
            velocities = [  self.velocity(d, times[i]) for i,d in  enumerate(distances) ]
            x_velocities = [  self.velocity(d, times[i]) for i,d in  enumerate(x_distances) ]        
            y_velocities = [  self.velocity(d, times[i]) for i,d in  enumerate(y_distances) ]   
            
            velocities_ref = [  self.velocity(d, times[i]) for i,d in  enumerate(distances_ref) ]
            x_velocities_ref = [  self.velocity(d, times[i]) for i,d in  enumerate(x_distances_ref) ]        
            y_velocities_ref = [  self.velocity(d, times[i]) for i,d in  enumerate(y_distances_ref) ]     
            
            df["elapsed_time"] = times[-original_size-1:-1]
            df["velocities"] = velocities[-original_size:]
            df["x_velocities"] = x_velocities[-original_size:]
            df["y_velocities"] = y_velocities[-original_size:]
            
            df["velocities_ref"] = velocities_ref[-original_size:]
            df["x_velocities_ref"] = x_velocities_ref[-original_size:]
            df["y_velocities_ref"] = y_velocities_ref[-original_size:]       
            
        if True:    
            aceleration = [ round(
                (velocities[i]-velocities[i+1])/times[i], 
                2) for i in range(len(velocities) - 1) ]
            x_aceleration = [ round(
                (x_velocities[i]-x_velocities[i+1])/times[i], 
                2) for i in range(len(x_velocities) - 1) ]
            y_aceleration = [ round(
                (y_velocities[i]-y_velocities[i+1])/times[i], 
                2) for i in range(len(y_velocities) - 1) ]
            
            aceleration_ref = [ round(
                (velocities_ref[i]-velocities_ref[i+1])/times[i], 
                2) for i in range(len(velocities_ref) - 1) ]
            x_aceleration_ref = [ round(
                (x_velocities_ref[i]-x_velocities_ref[i+1])/times[i], 
                2) for i in range(len(x_velocities_ref) - 1) ]
            y_aceleration_ref = [ round(
                (y_velocities_ref[i]-y_velocities_ref[i+1])/times[i], 
                2) for i in range(len(y_velocities_ref) - 1) ]   
            
            df["aceleration"] = aceleration[-original_size:]
            df["x_aceleration"] = x_aceleration[-original_size:]
            df["y_aceleration"] = y_aceleration[-original_size:]
            
            df["aceleration_ref"] = aceleration_ref[-original_size:]
            df["x_aceleration_ref"] = x_aceleration_ref[-original_size:]
            df["y_aceleration_ref"] = y_aceleration_ref[-original_size:]    
            
        if True:
            angle = [ self.angle(array[i], array[i+1]) for i in range(len(array) - 1)  ]
            angle_ref = [ self.angle(array[i], ref) for i in range(len(array))  ]
            
            df["angle"] = angle[-original_size:]
            df["angle_ref"] = angle_ref[-original_size:]
                    
        try:
            pass
        except Exception as e:
            print(e)

        kwargs["features"] = {
            "array": array[2:],
            
            "features_df": df,
            "features_np": df.values
        }

        return kwargs

    def velocity(self, distance, time_=None):
        if time_ is None or time_ == 0:
            time_ = self.TIME_FRAME
        return round(distance/time_, 2)

    @staticmethod 
    def angle(ref, point):
        delta_x = point[0] - ref[0]
        delta_y = point[1] - ref[1]
        
        angle_radian = math.atan2(delta_y, delta_x)
        angle_degrees = math.degrees(angle_radian)
        
        if delta_x < 0 and delta_y >= 0:
            angle_degrees += 180
        elif delta_x < 0 and delta_y < 0:
            angle_degrees -= 180        
        
        if angle_degrees < 0:
            angle_degrees += 360
            
        return round(angle_degrees, 2)
        

    @staticmethod 
    def distanceX(start, end):
        return end[0] - start[0]

    @staticmethod 
    def distanceY(start, end):
        return end[1] - start[1]
    
    @staticmethod 
    def distance(start, end):
        return round(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2), 2)

    @staticmethod 
    def get_quadrant(p, ref):
        if p[0] > ref[0] and p[1] > ref[1]:
            return 1
        elif p[0] < ref[0] and p[1] > ref[1]:
            return 2
        elif p[0] < ref[0] and p[1] < ref[1]:
            return 3
        elif p[0] > ref[0] and p[1] < ref[1]:
            return 4
        else:
            return 0