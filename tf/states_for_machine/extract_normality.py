import math, os


from states_for_machine.utils import Utils

area_calculate = Utils.area_calculate
procrustes_calculate = Utils.procrustes_calculate


from state_machine.State import State


class Extract_normality(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method to prepare data for train social cluster

        @type kwargs: dict
        @param kwargs: dict with data to process

        @rtype: dict
        @returns: dict with all_lines and all_names for train social cluster
        """  
        
        if not kwargs.get("tl_model"):
            raise Exception("No tl_model in kwargs")

        if not kwargs.get("tl_model").get("model"):
            raise Exception("No tl_model.model in kwargs")

        if not kwargs.get("path_save"):
            raise Exception("No path_save in kwargs")

        path_save = kwargs.get("path_save")
        model = kwargs.get("tl_model").get("model")

        all_areas = []
        all_procrustes = []
        
        for cluster in range(len(model.clusters)):
            areas = []
            procrustes_lineas = []
            brothers = model.get_brothers(cluster)
            for id_linea1 in range(len(brothers)):
                linea1 = brothers[id_linea1]
                
                areas_linea1 = [] # [0 for i in range(linea1+1)]
                procrustes_linea1 = []
                for id_linea2 in range(id_linea1+1, len(brothers), 1):
                    linea2 = brothers[id_linea2]
                    
                    area = area_calculate(  linea1, linea2, 50000 )
                    if not math.isnan(  area[0]   ):          
                        try:
                            procruste = procrustes_calculate(  linea1, linea2, 0.45 )        
                            procrustes_linea1.append(procruste[0])
                            
                            areas_linea1.append( area[0] )
            
                        except:
                            pass
                    
                areas.append(areas_linea1)
                procrustes_lineas.append(procrustes_linea1)
        
            all_areas.append( areas )
            all_procrustes.append( procrustes_lineas )

        tmp_summary = []
        for i, g in enumerate(all_procrustes):
            tmp_procrustes_lineas = []
            for p in all_procrustes[i]:
                tmp_procrustes_lineas += p

            tmp_areas = []
            for p in all_areas[i]:
                tmp_areas += p
            
            tmp_summary.append(
                (
                    min(tmp_procrustes_lineas), round(sum(tmp_procrustes_lineas)/len(tmp_procrustes_lineas), 4), max(tmp_procrustes_lineas),
                    min(tmp_areas), round(sum(tmp_areas)/len(tmp_areas), 3), max(tmp_areas)
                )
            )        

        if kwargs.get("verbose"):
            for g in range(len(tmp_summary)):
                print(g)
                print("\t", tmp_summary[g])

        model.normalization = tmp_summary

        model.save(path_save)

        kwargs["tl_model"]["model"] = model
        kwargs["tl_model"]["status_save"] = os.path.exists(path_save)        
        
        return kwargs

    