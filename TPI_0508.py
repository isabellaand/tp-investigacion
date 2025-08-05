import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------------
# CONFIGURACIÓN DEL ENTORNO
# -------------------------------
TAMANO_MAPA = 10
NUM_OBSTACULOS_LIVIANOS = 15
NUM_OBSTACULOS_PESADOS = 15
INICIO = (0, 0)
META = (9, 9)
TAMANO_POBLACION = 1000
GENERACIONES = 200
PROB_MUTACION = 0.2
MAX_PASOS = 30
MOVIMIENTOS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # derecha, abajo, izquierda, arriba

# -------------------------------
# FUNCIONES DEL MAPA
# -------------------------------
def crear_mapa(): # Crea un mapa con obstáculos livianos y pesados
    mapa = np.zeros((TAMANO_MAPA, TAMANO_MAPA), dtype=int) # 0: libre, 1: liviano, 2: pesado
    for _ in range(NUM_OBSTACULOS_LIVIANOS): # Crea obstáculos livianos
        x, y = np.random.randint(0, TAMANO_MAPA, 2)
        mapa[x][y] = 1
    for _ in range(NUM_OBSTACULOS_PESADOS): # Crea obstáculos pesados
        x, y = np.random.randint(0, TAMANO_MAPA, 2) 
        mapa[x][y] = 2
    mapa[INICIO] = 0
    mapa[META] = 0 
    return mapa

def mostrar_mapa(mapa, camino=None, titulo='Mapa'): # Muestra el mapa con el camino encontrado
    imagen = np.copy(mapa)
    if camino: # Si hay un camino, lo marca en el mapa
        for (x, y) in camino: # marca el camino
            if (x, y) != INICIO and (x, y) != META: # evita marcar inicio y meta
                imagen[x][y] = 3
    cmap = plt.colormaps['viridis'] # Define el mapa de colores
    plt.imshow(imagen, cmap=cmap)
    plt.title(titulo)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(ticks=[0, 1, 2, 3], label='0: libre, 1: liviano, 2: pesado, 3: camino')
    plt.scatter(INICIO[1], INICIO[0], marker='o', color='white', label='Inicio')
    plt.scatter(META[1], META[0], marker='*', color='red', label='Meta')
    plt.legend()
    plt.show()

# -------------------------------
# FUNCIONES DEL AG
# -------------------------------
def crear_individuo(): # Crea un individuo aleatorio, que es una secuencia de movimientos
    return [MOVIMIENTOS[random.randint(0, len(MOVIMIENTOS) - 1)] for _ in range(MAX_PASOS)] 

def decodificar_camino(individuo): # Decodifica un individuo a un camino en el mapa
    x, y = INICIO
    camino = [INICIO]
    for dx, dy in individuo: # mueve al individuo en el mapa
        nueva_x, nueva_y = x + dx, y + dy 
        if 0 <= nueva_x < TAMANO_MAPA and 0 <= nueva_y < TAMANO_MAPA: # verifica que el movimiento esté dentro del mapa
            x, y = nueva_x, nueva_y
            camino.append((x, y))
    return camino

def calcular_aptitud(individuo, mapa): # Calcula la aptitud de un individuo basado en el camino que recorre, penalizando repeticiones y obstáculos
    camino = decodificar_camino(individuo)
    puntaje = 0
    visitados = set()
    for (x, y) in camino: # verifica cada posición del camino
        if (x, y) in visitados:
            puntaje += 5
        visitados.add((x, y))
        if mapa[x][y] == 1:
            puntaje += 10
        elif mapa[x][y] == 2:
            puntaje += 50
    distancia = abs(x - META[0]) + abs(y - META[1]) # calcula la distancia al objetivo
    puntaje += distancia * 2
    return puntaje

def construir_ruleta_invertida(fitness): # Construye una ruleta invertida basada en la aptitud de los individuos
    total = sum(1 / (1 + f) for f in fitness) # suma las inversas de las aptitudes
    ruleta = []
    for i, f in enumerate(fitness): # calcula la probabilidad de cada individuo
        prob = (1 / (1 + f)) / total # normaliza la probabilidad
        ruleta.extend([i] * int(prob * 1000)) # esto es para aumentar la probabilidad de selección
    return ruleta

def seleccion_padres_ruleta(cromosomas, ruleta): # Selecciona dos padres de la ruleta invertida
    indice1 = random.randint(0, len(ruleta) - 1) 
    padre1 = cromosomas[ruleta[indice1]] # selecciona un padre al azar
    indice2 = random.randint(0, len(ruleta) - 1) 
    padre2 = cromosomas[ruleta[indice2]] # selecciona el segundo padre al azar
    while padre1 == padre2: # asegura que los padres sean diferentes
        indice2 = random.randint(0, len(ruleta) - 1)
        padre2 = cromosomas[ruleta[indice2]]
    return padre1, padre2

def cruzar(padre1, padre2): # Cruza dos padres para crear un hijo
    punto = random.randint(1, MAX_PASOS - 2) # selecciona un punto de cruce aleatorio
    return padre1[:punto] + padre2[punto:] # une los padres en el punto de cruce

def mutar(individuo): # Aplica una mutación aleatoria al individuo
    if random.random() < PROB_MUTACION: # verifica si se aplica la mutación
        idx = random.randint(0, MAX_PASOS - 1) # selecciona un índice aleatorio 
        individuo[idx] = MOVIMIENTOS[random.randint(0, len(MOVIMIENTOS) - 1)] # reemplaza el movimiento en el índice seleccionado
    return individuo

def algoritmo_genetico(mapa): # Algoritmo genético para encontrar el camino más corto
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)] # crea una población inicial de individuos
    mejor_individuo = None # inicializa el mejor individuo

    for _ in range(GENERACIONES): # itera por el número de generaciones
        puntajes = [calcular_aptitud(ind, mapa) for ind in poblacion] # calcula la aptitud de cada individuo
        ruleta = construir_ruleta_invertida(puntajes) # construye la ruleta invertida
        nueva_generacion = [] # crea una nueva generación de individuos

        # Elitismo
        mejor_individuo = min(poblacion, key=lambda ind: calcular_aptitud(ind, mapa)) # encuentra el mejor individuo de la generación actual
        nueva_generacion.append(mejor_individuo) # agrega el mejor individuo a la nueva generación

        while len(nueva_generacion) < TAMANO_POBLACION: # sigue llenando la nueva generación
            padre1, padre2 = seleccion_padres_ruleta(poblacion, ruleta) 
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo)
            nueva_generacion.append(hijo)

        poblacion = nueva_generacion

    mejor_final = min(poblacion, key=lambda ind: calcular_aptitud(ind, mapa)) # encuentra el mejor individuo de la última generación
    return decodificar_camino(mejor_final) 

# -------------------------------
# A* PARA COMPARACIÓN
# -------------------------------
def a_estrella(mapa): # Implementación del algoritmo A* para encontrar el camino óptimo
    def heuristica(a, b): # Calcula la heurística de Manhattan entre dos puntos
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) # distancia Manhattan

    cola = [(0 + heuristica(INICIO, META), 0, INICIO, [])] # Cola de prioridad para A*
    visitados = set()

    while cola: 
        cola.sort(key=lambda x: x[0]) # ordena la cola por costo estimado
        costo_estimado, costo_real, actual, camino = cola.pop(0) # saca el elemento con menor costo estimado
        # pop(0) extrae el primer nodo de la lista (el que tiene menor costo_estimado).

        if actual in visitados: 
            continue

        visitados.add(actual)
        camino = camino + [actual]

        if actual == META:
            return camino

        for dx, dy in MOVIMIENTOS: # mueve en las cuatro direcciones
            x, y = actual[0] + dx, actual[1] + dy
            if 0 <= x < TAMANO_MAPA and 0 <= y < TAMANO_MAPA and mapa[x][y] != 2:
                nuevo_costo = costo_real + (5 if mapa[x][y] == 1 else 1)
                cola.append((nuevo_costo + heuristica((x, y), META), nuevo_costo, (x, y), camino))
    return None

# -------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------- 
if __name__ == "__main__":
    mapa = crear_mapa()

    print("\nEjecutando algoritmo genético...")
    camino_ag = algoritmo_genetico(mapa)
    mostrar_mapa(mapa, camino_ag, titulo="Camino encontrado por AG + Elitismo")

    print("\nEjecutando A* para comparación...")
    camino_astar = a_estrella(mapa)
    mostrar_mapa(mapa, camino_astar, titulo="Camino óptimo encontrado por A*")

    print("\nResultados:")
    print(f"Longitud camino AG: {len(camino_ag)} pasos")
    print(f"Longitud camino A*: {len(camino_astar) if camino_astar else 'No encontrado'} pasos")
