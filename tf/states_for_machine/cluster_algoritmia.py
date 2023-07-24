from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict
from itertools import chain
from numba import njit

import warnings
warnings.filterwarnings('ignore')


class ClusterAlgoritmia:
    
    similar_lines_list = []
    trained = False
    
    def __init__(self, sensibility):
        """
        @type sensibility: float
        @param sensibility: sensibility for cluster
        """
        self.sensibility = sensibility
    
    def train(self, all_lines: list):
        """
        @type all_lines: list
        @param all_lines: list with all lines
        """
        self.all_lines = all_lines.copy()    
    
    def predict(self, new_vehicles: list) -> list:
        """
        @type new_vehicles: list
        @param new_vehicles: list with new vehicles

        @rtype: list
        @returns: list with predictions
        """
        predictions = []
        for new_vehicle in new_vehicles:
            
            all_lines_class = self.all_lines
            
            self.before = len(all_lines_class)

            if isinstance(new_vehicle, list):
                # add new vehicle to all_lines_class for predict after of train model
                all_lines_class.append(new_vehicle)
            
            after = len(all_lines_class)
            
            self.similar_lines_list = self.get_similars(all_lines_class, self.sensibility)
            
            self.trained = True

            if isinstance(new_vehicle, list):
                # if new_vehicle is exist and it is a list, extract the cluster of prediction
                for i, similar_list in enumerate(self.similar_lines_list):                    
                    if after in similar_list:                    
                        predictions.append(i)   
                    
        return predictions

    @staticmethod
    def _calculate_directed_hausdorff( line_a, line_b):
        """
        Calculate the directed Hausdorff distance between two lines.

        @type line_a: list
        @param line_a: list with points of line a

        @type line_b: list
        @param line_b: list with points of line b

        @rtype: float
        @returns: float with distance
        """
        return directed_hausdorff(line_a, line_b)[0]

    def _calcule_distance_Hausdorff(self, all_lines:list) -> dict:
        """
        Find the Hausdorff distance between each pair of lines.

        @type all_lines: list
        @param all_lines: list with all lines

        @rtype: dict
        @returns: dict with distances
        """
        distances_hausdorff = {}
    
        for i, line_a in enumerate(all_lines):

            for j, line_b in enumerate(all_lines):

                if i != j:  # Evitar comparar una línea consigo misma
                    distance = self._calculate_directed_hausdorff(line_a, line_b)
                    distances_hausdorff[(i, j)] = distance
        
        return distances_hausdorff
    
    @staticmethod
    @njit(nogil=True)
    def _find_similar_lines(pair:list, distance:list, umbral_similary:float = 45) -> list:
        """
        Find the similar lines.

        @type pair: list
        @param pair: list with pairs of lines

        @type distance: list
        @param distance: list with distances

        @type umbral_similary: float
        @param umbral_similary: float with umbral for similary

        @rtype: list
        @returns: list with similar lines
        """
        similar_lines = []
        for par, distance in zip(pair, distance):
            if distance < umbral_similary:
                line_a_idx, line_b_idx = par
                similar_lines.append((line_a_idx, line_b_idx))

        return similar_lines

    @staticmethod
    def _create_defaultdict(similar_pair:list) -> dict:
        """
        Create a defaultdict from a list of similar lines.

        @type similar_pair: list
        @param similar_pair: list with similar lines

        @rtype: dict
        @returns: dict with defaultdict
        """
        graph = defaultdict(list)
        for a, b in similar_pair:
            graph[a].append(b)
            graph[b].append(a)
        return graph

    @staticmethod
    @njit(nogil=True)
    def _show_similar_lines(similar_lines:list)-> dict:
        """
        show the similar lines

        @type similar_lines: list
        @param similar_lines: list with similar lines

        @rtype: dict
        @returns: dict with similar lines
        """
        similar_pairs = []
        for idx_a, idx_b in similar_lines:
            similar_pairs.append([idx_a + 1, idx_b + 1])
        return similar_pairs
    
    def _find_all_connected_components(self, graph:dict) -> list:
        """
        Find all connected components in a graph

        @type graph: dict
        @param graph: dict with graph

        @rtype: list
        @returns: list with all connected components
        """
        visited = set()
        connected_components = []
        for nodo in graph:
            if nodo not in visited:
                component = self.find_connected_component(nodo, graph)
                connected_components.append(component)
                visited.update(component)
        return connected_components
    
    def get_similars(self, all_lines:list, umbral_similary:float = 45) -> list:
        """
        Find the similar lines.

        @type all_lines: list
        @param all_lines: list with all lines

        @type umbral_similary: float
        @param umbral_similary: float with umbral for similary

        @rtype: list
        @returns: list with similar lines
        """
        distances_hausdorff = self._calcule_distance_Hausdorff(all_lines)
    
        pair = list(distances_hausdorff.keys())
        distance = list(distances_hausdorff.values())
        lineas_semejantes = self._find_similar_lines(pair, distance, umbral_similary)
    
        similar_pairs = self._show_similar_lines(lineas_semejantes)     
        graph = self._create_defaultdict(similar_pairs)
    
        conected_components = self._find_all_connected_components(graph)
    
        similar_lines_list = [list(component) for component in conected_components]
        
        return similar_lines_list

    @staticmethod
    def find_connected_component(nodo, graph:dict) -> set:
        """
        Find the connected component of a node in a graph

        @type nodo: int
        @param nodo: int with node

        @type graph: dict
        @param graph: dict with graph

        @rtype: set
        @returns: set with connected component
        """
        visited = set()
        component = set()
        stack = [nodo]

        while stack:
            _now = stack.pop()
            if _now not in visited:
                visited.add(_now)
                component.add(_now)
                stack.extend(graph[_now])

        return component

    def brothers(self, cluster:int) -> list:
        """
        Find the brothers of a cluster

        @type cluster: int
        @param cluster: int with cluster

        @rtype: list
        @returns: list with brothers
        """

        if self.trained:   # bool, especific if the model is trained     
            conten_cluster = []
            for d in self.similar_lines_list[cluster]:
                if d < len(self.all_lines):
                    conten_cluster.append(d-1)      
            return conten_cluster
        return []

    @property
    def get_clusters(self):
        """
        Find the clusters

        @rtype: list
        @returns: list with clusters
        """
        temp = []
        for cluster in range(len(self.similar_lines_list)):
            conten_cluster = self.brothers(cluster)
            temp.append(conten_cluster)
        return temp



