// Gerardo Molina Sáez y Bruno Jiménez Pertusa
// Práctica 2 - Ingeniería de los Computadores
// (Código original: Práctica 6 de ADA por Gerardo)

#include <fstream>
#include <string>
#include <limits>
#include <iostream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <chrono>
using namespace std;

// Variables globales
string file_name;

// Obtiene el mapa del fichero con el nombre pasado como parámetro
void obtener_mapa_fichero(string file_name, int &filas, int &columnas, vector<vector<int>> &mapa)
{

    // Abrimos el fichero
    ifstream fichero(file_name);

    // Leemos la primera fila (fila y columnas) y redimensionamos la matriz
    fichero >> filas >> columnas;
    mapa.resize(filas, vector<int>(columnas));

    // Rellenamos la matriz
    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            fichero >> mapa[i][j];
        }
    }
    fichero.close();
}

// Algoritmo recursivo sin almacén (ingenuo)
int maze_naive(vector<vector<int>> &mapa, int filas, int columnas, int pos_fila, int pos_columna)
{

    // Caso especial: nos salimos del laberinto o estamos en una casilla inaccesible
    if (pos_fila >= filas || pos_columna >= columnas || mapa[pos_fila][pos_columna] == 0)
    {
        return -1;
    }

    // Caso base: hemos llegado al destino
    if (pos_fila == filas - 1 && pos_columna == columnas - 1)
    {
        return 1;
    }

    // Caso recursivo, no estamos en el destino
    // Paso 1: recorrer los 3 movimientos posibles
    int mover_derecha = maze_naive(mapa, filas, columnas, pos_fila, pos_columna + 1);
    int mover_abajo = maze_naive(mapa, filas, columnas, pos_fila + 1, pos_columna);
    int mover_diagonal = maze_naive(mapa, filas, columnas, pos_fila + 1, pos_columna + 1);

    // Paso 2: comprobar si los movimientos son válidos (llegan a la meta)
    vector<int> caminos_posibles;
    if (mover_derecha != -1)
    {
        caminos_posibles.push_back(mover_derecha);
    }
    if (mover_abajo != -1)
    {
        caminos_posibles.push_back(mover_abajo);
    }
    if (mover_diagonal != -1)
    {
        caminos_posibles.push_back(mover_diagonal);
    }

    // Paso 3: seleccionamos el mejor movimiento (es decir, el camino más corto que llegue a la meta)
    if (caminos_posibles.empty())
    {
        return -1;
    }
    int mejor_caso = caminos_posibles[0];
    for (size_t i = 1; i < caminos_posibles.size(); i++)
    { // Uso size_t porque da warnring si uso int
        if (caminos_posibles[i] < mejor_caso)
        {
            mejor_caso = caminos_posibles[i];
        }
    }

    // Paso 4: devolvemos el mejor camino más el paso que acabamos de dar
    return mejor_caso + 1;
}

// Muestra qué se debe poner por terminal para que funcione correctamente
void show_usage()
{
    cout << "Usage: " << endl
         << "maze -f file" << endl;
}

// Valida y analiza los argumentos de la orden descrita por terminal
void check_argumentos(int argc, char *argv[])
{

    // Validación para pocos argumentos
    if (argc < 2)
    {
        show_usage();
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];

        // Validación del fichero
        if (arg == "-f")
        {
            i++;
            if (i >= argc)
            {
                cerr << "ERROR: missing filename. " << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
            file_name = argv[i];
            if (file_name.empty())
            {
                cerr << "ERROR: missing filename. " << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
            ifstream open_file(file_name);
            if (!open_file)
            {
                cerr << "ERROR: can't open file : " << file_name << "." << endl;
                show_usage();
                exit(EXIT_FAILURE);
            }
        }

        // Opción incorrecta
        else
        {
            cerr << "ERROR: unknown option " << arg << "." << endl;
            show_usage();
            exit(EXIT_FAILURE);
        }
    }
}

// Ejecución principal
int main(int argc, char *argv[])
{

    // Comprueba los argumentos de la orden
    check_argumentos(argc, argv);

    // Obtener el mapa del fichero
    int filas, columnas, resultado;
    vector<vector<int>> mapa;
    obtener_mapa_fichero(file_name, filas, columnas, mapa);

    // Iniciamos temporizador
    auto inicio = chrono::steady_clock::now();

    // Algoritmo ingenuo resuelve el problema 
    resultado = maze_naive(mapa, filas, columnas, 0, 0);
    auto fin = chrono::steady_clock::now();
    auto segundos = chrono::duration_cast<chrono::seconds>(fin - inicio);
    cout << segundos.count() << " segundos." << endl;
    cout << resultado << " movimientos." << endl;

    return 0;
}