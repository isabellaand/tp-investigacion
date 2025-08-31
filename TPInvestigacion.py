import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# -------------------------------
# CONFIGURACIÓN DEL ENTORNO
# -------------------------------
TAMANO_MAPA = 10
NUM_OBSTACULOS_LIVIANOS = 20#15
NUM_OBSTACULOS_PESADOS = 15
INICIO = (0, 0)
META = (9, 9)
TAMANO_POBLACION = 200
GENERACIONES = 400
PROB_MUTACION = 0.05
MAX_PASOS = 35
MOVIMIENTOS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # derecha, abajo, izquierda, arriba

# -------------------------------
# FUNCIONES DEL MAPA
# -------------------------------
def crear_mapa_alcanzable():
    while True:
        mapa = crear_mapa()
        camino_astar = a_estrella(mapa)
        if camino_astar is not None:
            return mapa

def crear_mapa():
    mapa = np.zeros((TAMANO_MAPA, TAMANO_MAPA), dtype=int)
    for _ in range(NUM_OBSTACULOS_LIVIANOS):
        x, y = np.random.randint(0, TAMANO_MAPA, 2)
        mapa[x][y] = 1
    for _ in range(NUM_OBSTACULOS_PESADOS):
        x, y = np.random.randint(0, TAMANO_MAPA, 2)
        mapa[x][y] = 2
    mapa[INICIO] = 0
    mapa[META] = 0
    return mapa

def mostrar_mapa(mapa, camino=None, titulo='Mapa'):
    imagen = np.array(mapa, dtype=int)

    if camino:
        for (x, y) in camino:
            if (x, y) != INICIO and (x, y) != META:
                if mapa[x][y] == 1:
                    imagen[x][y] = 4  # Camino sobre obstáculo liviano
                else:
                    imagen[x][y] = 3  # Camino libre

    # Definicion de colores
    colores = ['#440154', '#3b528b', '#21908d', '#fde725', '#fca50a']
    cmap = mcolors.ListedColormap(colores)

    # Normalización para valores discretos 0-4
    bounds = [0,1,2,3,4,5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(imagen, cmap=cmap, norm=norm)
    plt.title(titulo)
    plt.xticks([])
    plt.yticks([])

    cbar = plt.colorbar(ticks=[0,1,2,3,4])
    cbar.ax.set_yticklabels(['Libre (0)', 'Obstáculo liviano (1)', 'Obstáculo pesado (2)', 'Camino (3)', 'Camino sobre liviano (4)'])

    plt.scatter(INICIO[1], INICIO[0], marker='o', color='white', label='Inicio')
    plt.scatter(META[1], META[0], marker='*', color='red', label='Meta')
    plt.legend()
    plt.show()

# -------------------------------
# FUNCIONES DEL AG
# -------------------------------

def crear_individuo():
    return [MOVIMIENTOS[random.randint(0, len(MOVIMIENTOS) - 1)] for _ in range(MAX_PASOS)]

def decodificar_camino(individuo, mapa):
    x, y = INICIO
    camino = [INICIO]

    for dx, dy in individuo:
        nueva_x, nueva_y = x + dx, y + dy

        # Dentro del mapa
        if 0 <= nueva_x < TAMANO_MAPA and 0 <= nueva_y < TAMANO_MAPA:
            celda = mapa[nueva_x][nueva_y]

            if celda == 2:
                # Obstáculo pesado (2) -> no avanza
                continue
            else:
                # Libre (0) o liviano (1) -> avanza
                x, y = nueva_x, nueva_y
                camino.append((x, y))

                 # Sale del bucle si llega a la meta
                if (x, y) == META:
                    break

    return camino

def calcular_aptitud(individuo, mapa):
    camino = decodificar_camino(individuo, mapa)

    puntaje = 0
    visitados = set()
    # Penalizacion por volver a visitar una casilla y backtracking A-B-A
    for i, (x, y) in enumerate(camino):
        if (x, y) in visitados:
            puntaje += 10
        visitados.add((x, y))
        if i >= 2 and camino[i-2] == (x, y):
            puntaje += 15
        if mapa[x][y] == 1:
            puntaje += 20  # Obstaculo liviano -> penalizacion baja
        

    # Intentos de avanzar fuera del mapa
    x, y = INICIO
    fuera = 0
    for dx, dy in individuo:
        nx, ny = x + dx, y + dy
        if 0 <= nx < TAMANO_MAPA and 0 <= ny < TAMANO_MAPA:
            x, y = nx, ny
        else:
            fuera += 1
    puntaje += fuera * 8

    if META in camino:
        i = camino.index(META)
        puntaje -= 600          # Premio elevado por llegar a la meta
        puntaje += i            # Penalizacion por cantidad de pasos
    else:
        # Penalizacion por distancia hacia la meta
        x,y = camino[-1]
        dist = abs(x - META[0]) + abs(y - META[1])
        puntaje += dist * 8

    return puntaje

def construir_ruleta_invertida(fitness):
    total = sum(1 / (1 + f) for f in fitness)
    ruleta = []
    for i, f in enumerate(fitness):
        prob = (1 / (1 + f)) / total
        ruleta.extend([i] * int(prob * 1000))
    return ruleta

def seleccion_padres_ruleta(cromosomas, ruleta):
    indice1 = random.randint(0, len(ruleta) - 1)
    padre1 = cromosomas[ruleta[indice1]]
    indice2 = random.randint(0, len(ruleta) - 1)
    padre2 = cromosomas[ruleta[indice2]]
    while padre1 == padre2:
        indice2 = random.randint(0, len(ruleta) - 1)
        padre2 = cromosomas[ruleta[indice2]]
    return padre1, padre2

def cruzar(padre1, padre2):
    punto = random.randint(1, MAX_PASOS - 2)
    return padre1[:punto] + padre2[punto:]

def mutar(individuo):
    if random.random() < PROB_MUTACION:
        idx = random.randint(0, MAX_PASOS - 1)
        individuo[idx] = MOVIMIENTOS[random.randint(0, len(MOVIMIENTOS) - 1)]
    return individuo

def algoritmo_genetico(mapa):
    poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

    for _ in range(GENERACIONES):
        # Fitness de la población actual
        puntajes = [calcular_aptitud(ind, mapa) for ind in poblacion]

        # Elitismo -> conserva el mejor de la generación
        idx_mejor = min(range(len(poblacion)), key=lambda i: puntajes[i])
        mejor_individuo = poblacion[idx_mejor]

        # Construccion de ruleta invertida y nueva generación
        ruleta = construir_ruleta_invertida(puntajes)
        nueva_generacion = [mejor_individuo]  # elitismo

        while len(nueva_generacion) < TAMANO_POBLACION:
            padre1, padre2 = seleccion_padres_ruleta(poblacion, ruleta)
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo)
            nueva_generacion.append(hijo)

        poblacion = nueva_generacion

    # Datos de salida del mejor individuo final
    puntajes_final = [calcular_aptitud(ind, mapa) for ind in poblacion]
    idx_mejor_final = min(range(len(poblacion)), key=lambda i: puntajes_final[i])
    mejor_final = poblacion[idx_mejor_final]
    camino_final = decodificar_camino(mejor_final, mapa)
    aptitud_final = puntajes_final[idx_mejor_final]

    return mejor_final, idx_mejor_final, aptitud_final, camino_final

# -------------------------------
# A* PARA COMPARACIÓN
# -------------------------------

def a_estrella(mapa):
    def heuristica(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    cola = [(0 + heuristica(INICIO, META), 0, INICIO, [])]
    visitados = set()

    while cola:
        cola.sort(key=lambda x: x[0])
        costo_estimado, costo_real, actual, camino = cola.pop(0)

        if actual in visitados:
            continue

        visitados.add(actual)
        camino = camino + [actual]

        if actual == META:
            return camino

        for dx, dy in MOVIMIENTOS:
            x, y = actual[0] + dx, actual[1] + dy
            if 0 <= x < TAMANO_MAPA and 0 <= y < TAMANO_MAPA and mapa[x][y] != 2:
                nuevo_costo = costo_real + (5 if mapa[x][y] == 1 else 1)
                cola.append((nuevo_costo + heuristica((x, y), META), nuevo_costo, (x, y), camino))
    return None

# -------------------------------
# PROGRAMA PRINCIPAL
# -------------------------------

if __name__ == "__main__":
    mapa = crear_mapa_alcanzable()

    print("\nEjecutando algoritmo genético...")
    mejor_individuo, idx_mejor, aptitud_mejor, camino_ag = algoritmo_genetico(mapa)

    # Informacion de salida
    print("\n--- Resultado AG ---")
    print(f"Indice del individuo seleccionado (en la última generación): {idx_mejor}")
    print(f"Aptitud del mejor individuo: {aptitud_mejor}")
    print(f"Longitud del camino AG: {len(camino_ag)} pasos")
    print("Camino AG (secuencia de celdas):")
    print(camino_ag)
    print("Cromosoma del mejor individuo (movimientos dx,dy):")
    print(mejor_individuo)

    mostrar_mapa(mapa, camino_ag, titulo="Camino encontrado por AG + Elitismo")

    print("\nEjecutando A* para comparación...")
    camino_astar = a_estrella(mapa)
    mostrar_mapa(mapa, camino_astar, titulo="Camino óptimo encontrado por A*")

    print("\n--- Comparación ---")
    print(f"Longitud camino A*: {len(camino_astar) if camino_astar else 'No encontrado'} pasos")
