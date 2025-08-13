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
TAMANO_POBLACION = 100
GENERACIONES = 200
PROB_MUTACION = 0.2
MAX_PASOS = 50
MOVIMIENTOS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # derecha, abajo, izquierda, arriba

# -------------------------------
# FUNCIONES DEL MAPA
# -------------------------------
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

'''def mostrar_mapa(mapa, camino=None, titulo='Mapa'):
    imagen = np.copy(mapa)
    if camino:
        for (x, y) in camino:
            if (x, y) != INICIO and (x, y) != META:
                imagen[x][y] = 3
    cmap = plt.colormaps['viridis']
    plt.imshow(imagen, cmap=cmap)
    plt.title(titulo)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(ticks=[0, 1, 2, 3], label='0: libre, 1: liviano, 2: pesado, 3: camino')
    plt.scatter(INICIO[1], INICIO[0], marker='o', color='white', label='Inicio')
    plt.scatter(META[1], META[0], marker='*', color='red', label='Meta')
    plt.legend()
    plt.show()'''

def mostrar_mapa(mapa, camino=None, titulo='Mapa'):
    # Aseguramos tipo entero
    imagen = np.array(mapa, dtype=int)

    if camino:
        for (x, y) in camino:
            if (x, y) != INICIO and (x, y) != META:
                if mapa[x][y] == 1:
                    imagen[x][y] = 4  # Camino sobre obstáculo liviano
                else:
                    imagen[x][y] = 3  # Camino sobre libre o pesado

    # Definir colores
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

'''def decodificar_camino(individuo):
    x, y = INICIO
    camino = [INICIO]
    visitas = {(x, y): 1}

    for dx, dy in individuo:
        nueva_x, nueva_y = x + dx, y + dy

        if 0 <= nueva_x < TAMANO_MAPA and 0 <= nueva_y < TAMANO_MAPA:
            nueva_pos = (nueva_x, nueva_y)
            if visitas.get(nueva_pos, 0) < 1:
                x, y = nueva_pos
                camino.append((x, y))
                visitas[nueva_pos] = visitas.get(nueva_pos, 0) + 1
            # Si ya visitó 1 vez, simplemente no se actualiza x, y y se ignora el paso
        # Si se sale del mapa, también se ignora

    return camino'''

def decodificar_camino(individuo, mapa):
    x, y = INICIO
    camino = [INICIO]
    visitas = {(x, y): 1}

    for dx, dy in individuo:
        nueva_x, nueva_y = x + dx, y + dy

        # Dentro del mapa
        if 0 <= nueva_x < TAMANO_MAPA and 0 <= nueva_y < TAMANO_MAPA:
            celda = mapa[nueva_x][nueva_y]

            if celda == 2:
                # Obstáculo pesado: no se avanza
                continue
            else:
                # Libre (0) o liviano (1)
                nueva_pos = (nueva_x, nueva_y)
                if visitas.get(nueva_pos, 0) < 1:
                    x, y = nueva_pos
                    camino.append((x, y))
                    visitas[nueva_pos] = visitas.get(nueva_pos, 0) + 1
        # Si está fuera del mapa → ignorar paso

    return camino


def calcular_aptitud(individuo, mapa):
    camino = decodificar_camino(individuo, mapa)
    puntaje = 0
    visitados = set()

    for (x, y) in camino:
        if (x, y) in visitados:
            puntaje += 5  # penalización por repetir celda
        visitados.add((x, y))

        if mapa[x][y] == 1:
            puntaje += 50
        elif mapa[x][y] == 2:
            puntaje += 250

    # Penalización por intentos fuera del mapa
    x, y = INICIO
    fuera_de_mapa = 0
    for dx, dy in individuo:
        nueva_x, nueva_y = x + dx, y + dy
        if 0 <= nueva_x < TAMANO_MAPA and 0 <= nueva_y < TAMANO_MAPA:
            x, y = nueva_x, nueva_y
        else:
            fuera_de_mapa += 1

    puntaje += fuera_de_mapa * 10

    # Bono por llegar a la meta
    if camino[-1] == META:
        puntaje -= 100
    else:
        distancia = abs(camino[-1][0] - META[0]) + abs(camino[-1][1] - META[1])
        puntaje += distancia * 3

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
    mejor_individuo = None

    for _ in range(GENERACIONES):
        puntajes = [calcular_aptitud(ind, mapa) for ind in poblacion]
        ruleta = construir_ruleta_invertida(puntajes)
        nueva_generacion = []

        # Elitismo
        mejor_individuo = min(poblacion, key=lambda ind: calcular_aptitud(ind, mapa))
        nueva_generacion.append(mejor_individuo)

        while len(nueva_generacion) < TAMANO_POBLACION:
            padre1, padre2 = seleccion_padres_ruleta(poblacion, ruleta)
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo)
            nueva_generacion.append(hijo)

        poblacion = nueva_generacion

    mejor_final = min(poblacion, key=lambda ind: calcular_aptitud(ind, mapa))
    return decodificar_camino(mejor_final, mapa)

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
