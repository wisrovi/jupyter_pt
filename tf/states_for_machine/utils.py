from scipy.spatial import procrustes
from scipy import interpolate
import numpy as np



class Utils:
    @staticmethod
    def area_calculate(lista1, lista2, umbral=5000):
        # Obtener las coordenadas x e y de cada lista
        x1, y1 = zip(*lista1)
        x2, y2 = zip(*lista2)
        
        # Interpolar ambas listas para obtener funciones continuas
        f1 = interpolate.interp1d(x1, y1, kind='linear', fill_value='extrapolate')
        f2 = interpolate.interp1d(x2, y2, kind='linear', fill_value='extrapolate')
        
        # Crear una nueva lista de puntos para evaluar la diferencia entre las funciones
        x_new = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), 1000)
        
        # Evaluar las funciones en los nuevos puntos
        y1_new = f1(x_new)
        y2_new = f2(x_new)
        
        # Calcular la diferencia entre las dos funciones
        diferencia = y1_new - y2_new
        
        # Calcular el área entre las dos listas utilizando la integración numérica (método del trapecio)
        area = np.trapz(np.abs(diferencia), x_new)

        area = abs(area)
        
        return round(area, 3), area<=umbral

    @staticmethod
    def procrustes_calculate(lista1, lista2, umbral=1.15):
        # Convertir las listas de puntos a matrices numpy
        matriz1 = np.array(lista1)
        matriz2 = np.array(lista2)
        
        # Asegurarse de que ambas matrices tengan la misma cantidad de columnas (2 para coordenadas x e y)
        if matriz1.shape[1] != matriz2.shape[1]:
            print("Las matrices deben tener la misma cantidad de columnas para compararlas.")
        else:
            # Interpolar ambas listas para obtener la misma cantidad de puntos
            cantidad_puntos = 1000
            x1, y1 = matriz1[:, 0], matriz1[:, 1]
            x2, y2 = matriz2[:, 0], matriz2[:, 1]
            f1 = interpolate.interp1d(x1, y1, kind='linear', fill_value='extrapolate')
            f2 = interpolate.interp1d(x2, y2, kind='linear', fill_value='extrapolate')
            x_new = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), cantidad_puntos)
            matriz1_interp = np.column_stack((x_new, f1(x_new)))
            matriz2_interp = np.column_stack((x_new, f2(x_new)))
        
            # Aplicar el algoritmo de Procrustes para obtener la transformación óptima
            mtx1, mtx2, disparidad = procrustes(matriz1_interp, matriz2_interp)
        
            # Calcular la distancia entre las dos formas después de la transformación
            distancia_procrustes = np.sqrt(np.sum((mtx1 - mtx2) ** 2))
        
            
            #print("similares", distancia_procrustes<0.1)

            return round(distancia_procrustes, 5), distancia_procrustes<=umbral
