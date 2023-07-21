from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict
from itertools import chain
from numba import njit


import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Tiempo de ejecución de '{func.__name__}': {elapsed_time:.6f} segundos")
        return result
    return wrapper


class ClusterAlgoritmia:
    
    lineas_semejantes_lista = []
    trained = False
    
    def __init__(self, sensibility):
        self.sensibility = sensibility
    
    def train(self, all_lines):
        self.all_lines = all_lines.copy()    
    
    def predict(self, new_vehicles):
        predictions = []
        for new_vehicle in new_vehicles:
            
            all_lines_class = self.all_lines
            
            self.before = len(all_lines_class)

            if isinstance(new_vehicle, list):
                all_lines_class.append(new_vehicle)
            else:
                print("No se puede predecir ", new_vehicle)
            
            after = len(all_lines_class)
            
            self.lineas_semejantes_lista = self.get_semejantes(all_lines_class, self.sensibility)
            
            self.trained = True

            if isinstance(new_vehicle, list):
                for i, similar_list in enumerate(self.lineas_semejantes_lista):                    
                    if after in similar_list:                    
                        predictions.append(i)   
                    
        return predictions

    @staticmethod
    def _calculate_directed_hausdorff( line_a, line_b):
        return directed_hausdorff(line_a, line_b)[0]

    def _calcule_distance_Hausdorff(self, todas_las_lineas) -> dict:
        """
        Find the Hausdorff distance between each pair of lines.
        """
        distancias_hausdorff = {}
    
        # Find the Hausdorff distance between each pair of lines
        for i, linea_a in enumerate(todas_las_lineas):
            for j, linea_b in enumerate(todas_las_lineas):
                if i != j:  # Evitar comparar una línea consigo misma
                    distancia = self._calculate_directed_hausdorff(linea_a, linea_b)
                    distancias_hausdorff[(i, j)] = distancia
        
        return distancias_hausdorff
    
    @staticmethod
    @njit(nogil=True)
    def _find_similar_lines(par:list, distancia:list, umbral_similitud:float = 45) -> list:
        lineas_semejantes = []
        for par, distancia in zip(par, distancia):
            if distancia < umbral_similitud:
                linea_a_idx, linea_b_idx = par
                lineas_semejantes.append((linea_a_idx, linea_b_idx))

        return lineas_semejantes

    @staticmethod
    def _create_defaultdict(pares_semejantes:list) -> dict:
        # create a graph using a defaultdict dictionary
        grafo = defaultdict(list)
        for a, b in pares_semejantes:
            grafo[a].append(b)
            grafo[b].append(a)
        return grafo

    @staticmethod
    @njit(nogil=True)
    def _show_similar_lines(lineas_semejantes:list)-> dict:
        """
        show the similar lines
        """
        pares_semejantes = []
        for idx_a, idx_b in lineas_semejantes:
            # print(f"\tLínea {idx_a + 1} es semejante a Línea {idx_b + 1}")
            pares_semejantes.append([idx_a + 1, idx_b + 1])
        return pares_semejantes
    
    def _find_all_connected_components(self, grafo):
        """
        Find all connected components in a graph
        """
        visitados = set()
        componentes_conectadas = []
        for nodo in grafo:
            if nodo not in visitados:
                componente = self.encontrar_componente_conectada(nodo, grafo)
                componentes_conectadas.append(componente)
                visitados.update(componente)
        return componentes_conectadas
    
    def get_semejantes(self, todas_las_lineas, umbral_similitud = 45):
        distancias_hausdorff = self._calcule_distance_Hausdorff(todas_las_lineas)
    
        par = list(distancias_hausdorff.keys())
        distancia = list(distancias_hausdorff.values())
        lineas_semejantes = self._find_similar_lines(par, distancia, umbral_similitud)
    
        pares_semejantes = self._show_similar_lines(lineas_semejantes)     
        grafo = self._create_defaultdict(pares_semejantes)

    
        componentes_conectadas = self._find_all_connected_components(grafo)
    
        lineas_semejantes_lista = [list(componente) for componente in componentes_conectadas]
        
        return lineas_semejantes_lista

    # Función para encontrar la componente conectada desde un nodo dado
    @staticmethod
    def encontrar_componente_conectada(nodo, grafo:dict):
        visitados = set()
        componente = set()
        pila = [nodo]

        while pila:
            actual = pila.pop()
            if actual not in visitados:
                visitados.add(actual)
                componente.add(actual)
                pila.extend(grafo[actual])

        return componente

    def brothers(self, cluster:int):
        if self.trained:   # bool, especific if the model is trained     
            conten_cluster = []
            for d in self.lineas_semejantes_lista[cluster]:
                if d < len(self.all_lines):
                    conten_cluster.append(d-1)      
            return conten_cluster
        return []

    @property
    @njit(nogil=True)  # nogil=True for the GIL to be released
    def get_clusters(self):
        temp = []
        for cluster in range(len(self.lineas_semejantes_lista)):
            conten_cluster = self.brothers(cluster)
            temp.append(conten_cluster)
        return temp



