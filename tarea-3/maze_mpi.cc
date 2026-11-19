#include <fstream>
#include <string>
#include <limits>
#include <iostream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <climits> // MPI: para INT_MAX
#include <mpi.h> // MPI: biblioteca de paso de mensajes
using namespace std;

// Variables globales
string file_name;

// Obtiene el mapa del fichero con el nombre pasado como parámetro
// CAMBIO PARA PARALELIZAR: la matriz se guarda en un único vector, fila a fila: mapa[i * columnas + j]
void obtener_mapa_fichero(string file_name, int &filas, int &columnas, vector<int> &mapa) {

    // Abrimos el fichero
    ifstream fichero(file_name);

    // Leemos la primera fila (fila y columnas) y redimensionamos el vector
    fichero >> filas >> columnas;
    mapa.resize(filas * columnas);

    // Rellenamos la matriz
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            fichero >> mapa[i * columnas + j];
        }
    }
    fichero.close();
}

// Algoritmo recursivo sin almacén (ingenuo)
int maze_naive(vector<int> &mapa, int filas, int columnas, int pos_fila, int pos_columna) {

    // Caso especial: nos salimos del laberinto o estamos en una casilla inaccesible
    if (pos_fila >= filas || pos_columna >= columnas || mapa[pos_fila * columnas + pos_columna] == 0) {
        return -1;
    }

    // Caso base: hemos llegado al destino
    if (pos_fila == filas - 1 && pos_columna == columnas - 1) {
        return 1;
    }

    // Caso recursivo, no estamos en el destino
    // Paso 1: recorrer los 3 movimientos posibles
    int mover_derecha = maze_naive(mapa, filas, columnas, pos_fila, pos_columna + 1);
    int mover_abajo = maze_naive(mapa, filas, columnas, pos_fila + 1, pos_columna);
    int mover_diagonal = maze_naive(mapa, filas, columnas, pos_fila + 1, pos_columna + 1);

    // Paso 2: comprobar si los movimientos son válidos (llegan a la meta)
    vector<int> caminos_posibles;
    if (mover_derecha != -1) {
        caminos_posibles.push_back(mover_derecha);
    }
    if (mover_abajo != -1) {
        caminos_posibles.push_back(mover_abajo);
    }
    if (mover_diagonal != -1) {
        caminos_posibles.push_back(mover_diagonal);
    }

    // Paso 3: seleccionamos el mejor movimiento (es decir, el camino más corto que llegue a la meta)
    if (caminos_posibles.empty()) {
        return -1;
    }
    int mejor_caso = caminos_posibles[0];
    for (size_t i = 1; i < caminos_posibles.size(); i++) { // Uso size_t porque da warnring si uso int
        if (caminos_posibles[i] < mejor_caso) {
            mejor_caso = caminos_posibles[i];
        }
    }

    // Paso 4: devolvemos el mejor camino más el paso que acabamos de dar
    return mejor_caso + 1;
}

// MPI: NUEVA. Expande el árbol de recursión 'profundidad' niveles y devuelve la
// lista de subproblemas independientes. Cada tarea son 3 enteros consecutivos:
// (fila, columna, pasos), donde 'pasos' = casillas recorridas hasta esa celda (incluida).
// Todos los procesos la ejecutan y obtienen la misma lista, así que no hay que enviarla.
vector<int> generar_tareas(vector<int> &mapa, int filas, int columnas, int profundidad)
{
    vector<int> tareas;
    if (mapa[0] != 0) {
        tareas = {0, 0, 1};
    }

    for (int nivel = 0; nivel < profundidad; nivel++) {
        vector<int> siguientes;
        for (size_t i = 0; i < tareas.size(); i += 3)
        {
            int f = tareas[i], c = tareas[i + 1], pasos = tareas[i + 2];

            // Si ya estamos en la meta, la tarea se conserva tal cual
            if (f == filas - 1 && c == columnas - 1)
            {
                siguientes.insert(siguientes.end(), {f, c, pasos});
                continue;
            }

            // Los mismos 3 movimientos de maze_naive: derecha, abajo y diagonal
            int nf[3] = {f, f + 1, f + 1};
            int nc[3] = {c + 1, c, c + 1};
            for (int k = 0; k < 3; k++)
            {
                if (nf[k] < filas && nc[k] < columnas && mapa[nf[k] * columnas + nc[k]] != 0)
                {
                    siguientes.insert(siguientes.end(), {nf[k], nc[k], pasos + 1});
                }
            }
        }
        tareas = siguientes;
    }
    return tareas;
}

// Muestra qué se debe poner por terminal para que funcione correctamente
void show_usage() {
    cout << "Usage: " << endl
         << "mpirun -np N maze_mpi -f file [-d profundidad]" << endl; // MPI: nueva opción -d
}

// Valida y analiza los argumentos de la orden descrita por terminal
void check_argumentos(int argc, char *argv[]) {

    // Validación para pocos argumentos
    if (argc < 2) {
        show_usage();
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        // Validación del fichero
        if (arg == "-f") {
            i++;
            if (i >= argc) {
                cerr << "ERROR: missing filename. " << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
            file_name = argv[i];
            if (file_name.empty()) {
                cerr << "ERROR: missing filename. " << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
            ifstream open_file(file_name);
            if (!open_file) {
                cerr << "ERROR: can't open file : " << file_name << "." << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
        }

        // Opción incorrecta
        else {
            cerr << "ERROR: unknown option " << arg << "." << endl;
            show_usage();
            exit(EXIT_FAILURE);
        }
    }
}

// Ejecución principal
int main(int argc, char *argv[]) {

    // PASO 1: Arrancamos MPI (size=nºprocesos, rank=identificador)
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // PASO 2: Obtenemos el mapa del fichero (el proceso principal tiene rank=0)
    int filas, columnas, profundidad = 5, resultado; // Profundidad le he metido 6 por defecto
    vector<int> mapa;
    if (rank == 0) {
        check_argumentos(argc, argv);
        obtener_mapa_fichero(file_name, filas, columnas, mapa);
    }

    // PASO 3: El proceso 0 envía al resto de procesos: dimensiones, profundidad y mapa
    MPI_Bcast(&filas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&profundidad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mapa.resize(filas * columnas); // Cada proceso reserva en memoria para el mapa
    MPI_Bcast(mapa.data(), filas * columnas, MPI_INT, 0, MPI_COMM_WORLD); // VECTOR: un solo envío

    // PASO 4: Sincroinzar para que todos empiecen a la vez y medir tiempo (MPI_Wtime en vez de Chrono)
    MPI_Barrier(MPI_COMM_WORLD);
    double inicio = MPI_Wtime();

    // PASO 5: en lugar de una única llamada maze_naive(0, 0), creamos las tareas...
    vector<int> tareas = generar_tareas(mapa, filas, columnas, profundidad);

    // ...y cada proceso resuelve solo las suyas (reparto cíclico: rank, rank+size, ...)
    int mejor_local = INT_MAX;
    for (size_t i = 3 * rank; i < tareas.size(); i += 3 * size) {
        int camino = maze_naive(mapa, filas, columnas, tareas[i], tareas[i + 1]);
        if (camino != -1) {
            // pasos ya dados hasta la tarea (sin contar la propia casilla) + resto del camino
            mejor_local = min(mejor_local, tareas[i + 2] - 1 + camino);
        }
    }

    // PASO 6: Cogemos el mejor camino global (el mínimo de los resultados de todos los procesos)
    int mejor_global;
    MPI_Reduce(&mejor_local, &mejor_global, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    resultado = (mejor_global == INT_MAX) ? -1 : mejor_global;

    // PASO 7: Obtenemos el tiempo (el ultimo en acabar es el tiempo total) y lo imprime el proceso 0
    double tiempo = MPI_Wtime() - inicio, tiempo_max;
    MPI_Reduce(&tiempo, &tiempo_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << tiempo_max << " segundos (" << size << " procesos, profundidad "
             << profundidad << ")." << endl;
        cout << resultado << " movimientos." << endl;
    }
    MPI_Finalize();
    return 0;
}