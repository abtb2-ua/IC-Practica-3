// knn_mpi.cc
// Versión paralela (MPI) del ajuste de K y evaluación final de un kNN sobre MNIST.
//
// Uso:
//   mpirun -np P ./cv_mpi [nTune folds maxK nTest mode evalProcs order chunk]
//     nTune      muestras usadas en el ajuste de K              (def. 20000)
//     folds      número de folds                                (def. 5)
//     maxK       se prueba K = 1..maxK                          (def. 10)
//     nTest      muestras de test en la evaluación final        (def. 10000)
//     mode       data | spec                                    (def. data)
//     evalProcs  procesos del grupo de evaluación (solo spec)   (def. 0 = automático)
//     order      asc | desc: orden en que el ajuste recorre K   (def. asc)
//     chunk      muestras entre sondeos de mensajes (solo spec) (def. 10)
//
// Modos:
//   data  Paralelismo de datos. Todos los procesos hacen el ajuste (cada uno una parte
//         de las muestras de validación) y después la evaluación final.
//   spec  Paralelismo de datos + funcional + ejecución especulativa. Un grupo de procesos
//         ajusta K mientras otro grupo evalúa el test con el mejor K conocido hasta el
//         momento, reiniciando si el ajuste encuentra uno mejor.

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

using namespace knn;
using namespace mnist;

namespace
{
const int ROOT = 0;         // Proceso (rank global) que imprime resultados.
const int TAG_K = 100;      // Mensajes del ajuste hacia la evaluación (NEWK / FIN).
const int TAG_RESULT = 101; // Resultado final de la evaluación hacia ROOT.
const int MSG_NEWK = 0;     // Hay un nuevo mejor K.
const int MSG_FIN = 1;      // El ajuste ha terminado: este es el K definitivo.

struct Params
{
    std::size_t nTune = 20000;
    unsigned int folds = 5;
    unsigned int maxK = 10;
    std::size_t nTest = 10000;
    bool speculative = false;
    int evalProcs = 0;
    bool descending = false;
    std::size_t chunk = 10;
};

// Medidas de un proceso (para repartir tiempo entre cómputo y comunicación).
struct Stats
{
    int role = 0;            // 0 = todos (data), 1 = ajuste, 2 = evaluación.
    double compute = 0.0;    // Tiempo de trabajo local.
    double comm = 0.0;       // Tiempo dentro de llamadas MPI (comunicación + esperas).
    double finish = 0.0;     // Instante (desde el inicio) en que termina su trabajo.
    int restarts = 0;        // Reinicios por K especulativo invalidado.
    long long wasted = 0;    // Muestras de test clasificadas y descartadas.
};

struct Outcome
{
    int bestK = 0;
    double accuracy = 0.0;
    double tuneTime = 0.0;
    double evalTime = 0.0;
};

bool parseArgs(int argc, char** argv, int worldSize, Params& p)
{
    try
    {
        if (argc > 1) p.nTune = std::stoul(argv[1]);
        if (argc > 2) p.folds = std::stoul(argv[2]);
        if (argc > 3) p.maxK = std::stoul(argv[3]);
        if (argc > 4) p.nTest = std::stoul(argv[4]);
        if (argc > 5)
        {
            std::string m = argv[5];
            if (m == "spec") p.speculative = true;
            else if (m != "data") return false;
        }
        if (argc > 6) p.evalProcs = std::stoi(argv[6]);
        if (argc > 7)
        {
            std::string o = argv[7];
            if (o == "desc") p.descending = true;
            else if (o != "asc") return false;
        }
        if (argc > 8) p.chunk = std::stoul(argv[8]);
    }
    catch (const std::exception&)
    {
        return false;
    }

    if (p.folds < 2 || p.maxK < 1 || p.nTune < p.folds || p.nTest < 1 || p.chunk < 1)
        return false;

    if (p.speculative)
    {
        if (worldSize < 2) return false;
        // evalProcs <= 0: automático. Se calcula en main() con autoEvalProcs(), una vez
        // conocidos los tamaños reales de los conjuntos.
        if (p.evalProcs > 0) p.evalProcs = std::min(p.evalProcs, worldSize - 1);
    }
    return true;
}

// Fracción del trabajo que corresponde a la evaluación final, estimada por número de
// distancias calculadas (coste dominante de kNN):
//   ajuste:     maxK * sum_folds |val| * |train_fold| = maxK * nTune * (nTune - nTune/folds)
//   evaluación: nTest * nTrain
// Con los valores por defecto (20000, 5, 10, 10000, 60000) sale ~15.8 %, coherente con el
// ~16.5 % medido con gprof; para otros tamaños cambia mucho (p. ej. 5000/2000 -> ~37 %).
double evalFraction(std::size_t nTune, unsigned int folds, unsigned int maxK,
                    std::size_t nTest, std::size_t nTrain)
{
    const double t = static_cast<double>(nTune);
    const double tune = maxK * t * (t - t / folds);
    const double eval = static_cast<double>(nTest) * static_cast<double>(nTrain);
    return eval / (eval + tune);
}

// Procesos de evaluación proporcionales a su carga: al menos 1 y dejando al menos 1 al ajuste.
int autoEvalProcs(double fraction, int worldSize)
{
    int n = static_cast<int>(std::lround(fraction * worldSize));
    return std::min(std::max(1, n), worldSize - 1);
}

bool loadDataset(std::vector<TSample>& train, std::vector<TSample>& test)
{
    if (!loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train))
        return false;
    return loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test);
}

// Reparto por bloques: el proceso r de p recibe [begin, end) de n elementos.
void blockRange(std::size_t n, int p, int r, std::size_t& begin, std::size_t& end)
{
    begin = n * r / p;
    end = n * (r + 1) / p;
}

// Valores de K en el orden en que el ajuste los recorre.
std::vector<int> kSequence(unsigned int maxK, bool descending)
{
    std::vector<int> ks;
    for (unsigned int k = 1; k <= maxK; k++) ks.push_back(static_cast<int>(k));
    if (descending) std::reverse(ks.begin(), ks.end());
    return ks;
}

// Criterio de mejora. En orden ascendente equivale al '>' estricto del secuencial;
// en orden descendente el desempate (menor K) da el mismo K final que el secuencial.
bool better(double acc, int k, double bestAcc, int bestK)
{
    return acc > bestAcc || (acc == bestAcc && k < bestK);
}

// Mismo reparto de folds que CrossValidation::splitDataset.
void splitFold(const std::vector<TSample>& data, unsigned int folds, unsigned int fold,
               std::vector<TSample>& train, std::vector<TSample>& val)
{
    std::size_t start = data.size() * fold / folds;
    std::size_t end = data.size() * (fold + 1) / folds;
    for (std::size_t i = 0; i < data.size(); i++)
    {
        if (i < start || i >= end) train.push_back(data[i]);
        else val.push_back(data[i]);
    }
}

// Clasifica set[begin, end) y devuelve cuántas etiquetas acierta.
long long countMatches(const KNN& clf, const std::vector<TSample>& set,
                       std::size_t begin, std::size_t end)
{
    long long matches = 0;
    for (std::size_t i = begin; i < end; i++)
        if (clf.classify(set[i]) == set[i].label) matches++;
    return matches;
}

// Validación cruzada de un K repartida entre los procesos del comunicador 'comm'.
// Cada proceso clasifica su bloque de las muestras de validación de cada fold y los
// aciertos se suman con MPI_Allreduce. Todos los procesos del comunicador obtienen
// la misma precisión (misma aritmética, mismo orden de folds que el secuencial).
double crossValidateK(const std::vector<TSample>& tuneSet, const Params& p, int k,
                      MPI_Comm comm, Stats& st)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double accuracy = 0.0;
    for (unsigned int fold = 0; fold < p.folds; fold++)
    {
        double t0 = MPI_Wtime();
        std::vector<TSample> train, val;
        splitFold(tuneSet, p.folds, fold, train, val);
        KNN clf(train, k);
        std::size_t b, e;
        blockRange(val.size(), size, rank, b, e);
        long long local = countMatches(clf, val, b, e);
        double t1 = MPI_Wtime();

        long long global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, comm);
        double t2 = MPI_Wtime();

        st.compute += t1 - t0;
        st.comm += t2 - t1;
        accuracy += global / static_cast<double>(val.size());
    }
    return accuracy / p.folds;
}

// ---------------------------------------------------------------------------
// Modo "data": paralelismo de datos puro.
// ---------------------------------------------------------------------------
void runDataParallel(const std::vector<TSample>& tuneSet, const std::vector<TSample>& train,
                     const std::vector<TSample>& test, const Params& p, double t0,
                     Stats& st, Outcome& out)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    st.role = 0;

    // Fase 1: ajuste de K (todos los procesos colaboran en cada validación cruzada).
    double tuneStart = MPI_Wtime();
    int bestK = 1;
    double bestAcc = 0.0;
    for (int k : kSequence(p.maxK, p.descending))
    {
        double ks = MPI_Wtime();
        double acc = crossValidateK(tuneSet, p, k, MPI_COMM_WORLD, st);
        if (rank == ROOT)
            std::cout << "K: " << k << " Accuracy: " << acc * 100 << "% completed in "
                      << MPI_Wtime() - ks << " seconds" << std::endl;
        if (better(acc, k, bestAcc, bestK))
        {
            bestAcc = acc;
            bestK = k;
        }
    }
    out.bestK = bestK;
    out.tuneTime = MPI_Wtime() - tuneStart;

    // Fase 2: evaluación final con el K elegido (dependencia: necesita bestK).
    double evalStart = MPI_Wtime();
    KNN clf(train, bestK);
    std::size_t b, e;
    blockRange(test.size(), size, rank, b, e);
    long long local = countMatches(clf, test, b, e);
    double t1 = MPI_Wtime();

    long long total = 0;
    MPI_Reduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, ROOT, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    st.compute += t1 - evalStart;
    st.comm += t2 - t1;
    out.accuracy = total / static_cast<double>(test.size());
    out.evalTime = t2 - evalStart;
    st.finish = t2 - t0;
}

// ---------------------------------------------------------------------------
// Modo "spec": paralelismo funcional + ejecución especulativa.
// ---------------------------------------------------------------------------

// El proceso raíz del ajuste avisa a cada proceso de evaluación. Se usan envíos punto a punto
// (no un Bcast) porque los receptores los atienden de forma asíncrona, entre bloques de trabajo.
void sendToEval(int type, int k, int firstEval, int worldSize, Stats& st)
{
    int msg[2] = {type, k};
    double t = MPI_Wtime();
    for (int r = firstEval; r < worldSize; r++)
        MPI_Send(msg, 2, MPI_INT, r, TAG_K, MPI_COMM_WORLD);
    st.comm += MPI_Wtime() - t;
}

void runSpeculative(const std::vector<TSample>& tuneSet, const std::vector<TSample>& train,
                    const std::vector<TSample>& test, const Params& p, double t0,
                    Stats& st, Outcome& out)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int tuneProcs = size - p.evalProcs;
    const bool isTune = rank < tuneProcs;
    st.role = isTune ? 1 : 2;

    // Dos grupos: ranks [0, tuneProcs) ajustan K y [tuneProcs, size) evalúan el test.
    MPI_Comm group;
    MPI_Comm_split(MPI_COMM_WORLD, isTune ? 0 : 1, rank, &group);
    int grank, gsize;
    MPI_Comm_rank(group, &grank);
    MPI_Comm_size(group, &gsize);

    const std::vector<int> ks = kSequence(p.maxK, p.descending);

    if (isTune)
    {
        // ---------------- Grupo de ajuste ----------------
        double tuneStart = MPI_Wtime();
        int bestK = 1;
        double bestAcc = 0.0;
        for (int k : ks)
        {
            double kStart = MPI_Wtime();
            double acc = crossValidateK(tuneSet, p, k, group, st);
            if (grank == 0)
                std::cout << "K: " << k << " Accuracy: " << acc * 100 << "% completed in "
                          << MPI_Wtime() - kStart << " seconds" << std::endl;
            if (better(acc, k, bestAcc, bestK))
            {
                bestAcc = acc;
                bestK = k;
                // Nuevo mejor K: se comunica a la evaluación para que reinicie si hace falta.
                if (grank == 0) sendToEval(MSG_NEWK, k, tuneProcs, size, st);
            }
        }
        // Fin del ajuste: K definitivo.
        if (grank == 0) sendToEval(MSG_FIN, bestK, tuneProcs, size, st);
        out.bestK = bestK;
        out.tuneTime = MPI_Wtime() - tuneStart;
        st.finish = MPI_Wtime() - t0;

        if (rank == ROOT)
        {
            // Recoge el resultado de la evaluación (espera = tiempo ocioso, no comunicación).
            double res[2];
            MPI_Recv(res, 2, MPI_DOUBLE, tuneProcs, TAG_RESULT, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            out.accuracy = res[0];
            out.evalTime = res[1];
            st.finish = MPI_Wtime() - t0;
        }
    }
    else
    {
        // ---------------- Grupo de evaluación ----------------
        double evalStart = MPI_Wtime();
        std::size_t b, e;
        blockRange(test.size(), gsize, grank, b, e);

        int curK = ks[0];     // Hipótesis inicial: el primer K que recorre el ajuste.
        int latestK = curK;   // Último K comunicado por el ajuste.
        bool fin = false;     // ¿Ha llegado el mensaje FIN?

        double c0 = MPI_Wtime();
        std::unique_ptr<KNN> clf(new KNN(train, curK));
        st.compute += MPI_Wtime() - c0;

        std::size_t pos = b;
        long long matches = 0;

        auto handle = [&](const int* msg)
        {
            latestK = msg[1];
            if (msg[0] == MSG_FIN) fin = true;
        };

        for (;;)
        {
            // 1. Atender mensajes pendientes sin bloquearse (MPI_Iprobe + MPI_Recv).
            double m0 = MPI_Wtime();
            for (;;)
            {
                int flag = 0;
                MPI_Iprobe(ROOT, TAG_K, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                if (!flag) break;
                int msg[2];
                MPI_Recv(msg, 2, MPI_INT, ROOT, TAG_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                handle(msg);
            }
            st.comm += MPI_Wtime() - m0;

            // 2. Si el K especulado ya no es el mejor conocido: invalidar y reiniciar.
            if (latestK != curK)
            {
                if (grank == 0)
                    std::cout << "[eval] reinicio: K " << curK << " -> " << latestK
                              << " (" << pos - b << " muestras descartadas en este proceso)"
                              << std::endl;
                st.restarts++;
                st.wasted += static_cast<long long>(pos - b);
                curK = latestK;
                double r0 = MPI_Wtime();
                clf.reset(new KNN(train, curK));
                st.compute += MPI_Wtime() - r0;
                pos = b;
                matches = 0;
            }

            // 3. Trabajo: un bloque de muestras con el K actual.
            if (pos < e)
            {
                std::size_t stop = std::min(pos + p.chunk, e);
                double w0 = MPI_Wtime();
                matches += countMatches(*clf, test, pos, stop);
                st.compute += MPI_Wtime() - w0;
                pos = stop;
            }
            else if (fin)
            {
                break; // Bloque completo con el K definitivo (curK == latestK).
            }
            else
            {
                // Bloque terminado pero el ajuste aún no ha acabado: esperar mensaje.
                double w0 = MPI_Wtime();
                int msg[2];
                MPI_Recv(msg, 2, MPI_INT, ROOT, TAG_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                handle(msg);
                st.comm += MPI_Wtime() - w0;
            }
        }

        // Suma de aciertos de todo el grupo de evaluación.
        double r0 = MPI_Wtime();
        long long total = 0;
        MPI_Reduce(&matches, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, group);
        st.comm += MPI_Wtime() - r0;

        if (grank == 0)
        {
            double res[2] = {total / static_cast<double>(test.size()),
                             MPI_Wtime() - evalStart};
            MPI_Send(res, 2, MPI_DOUBLE, ROOT, TAG_RESULT, MPI_COMM_WORLD);
        }
        st.finish = MPI_Wtime() - t0;
    }

    MPI_Comm_free(&group);
}

void printReport(const Params& p, int size, const Outcome& out, double totalTime,
                 const std::vector<double>& all)
{
    std::cout << "Best k: " << out.bestK << " found in " << out.tuneTime << " seconds" << std::endl;
    std::cout << "Test accuracy: " << out.accuracy << " completed in " << out.evalTime
              << " seconds" << std::endl;
    std::cout << "Total elapsed time " << totalTime << " seconds" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nrank  rol      computo  comunic.  termina  reinicios  descartadas" << std::endl;
    double sumCompute = 0.0, sumComm = 0.0;
    long long sumWasted = 0;
    const char* roles[] = {"todos", "ajuste", "eval"};
    for (int r = 0; r < size; r++)
    {
        const double* a = &all[6 * r];
        std::cout << std::setw(4) << r << "  " << std::setw(6) << std::left << roles[(int)a[0]]
                  << std::right << std::setw(10) << a[1] << std::setw(10) << a[2]
                  << std::setw(9) << a[3] << std::setw(11) << (int)a[4]
                  << std::setw(13) << (long long)a[5] << std::endl;
        sumCompute += a[1];
        sumComm += a[2];
        sumWasted += (long long)a[5];
    }
    std::cout << std::defaultfloat << std::setprecision(6); // setprecision(2) es persistente: restaurar.

    // Línea fácil de procesar para las gráficas del paso e.
    std::cout << "RESULT," << (p.speculative ? "spec" : "data") << "," << size << ","
              << (p.speculative ? p.evalProcs : 0) << "," << p.nTune << "," << p.folds << ","
              << p.maxK << "," << p.nTest << "," << (p.descending ? "desc" : "asc") << ","
              << out.bestK << "," << out.accuracy << "," << totalTime << "," << out.tuneTime
              << "," << out.evalTime << "," << sumCompute << "," << sumComm << ","
              << sumWasted << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Params p;
    if (!parseArgs(argc, argv, size, p))
    {
        if (rank == ROOT)
            std::cerr << "Uso: mpirun -np P " << argv[0]
                      << " [nTune folds maxK nTest data|spec evalProcs asc|desc chunk]\n"
                         "(el modo spec necesita al menos 2 procesos)" << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Todos los procesos arrancan a la vez y miden desde aquí (incluye la carga de datos,
    // igual que 'Total elapsed time' del secuencial).
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Cada proceso carga MNIST por su cuenta: así no hay que serializar TSample.
    std::vector<TSample> train, test;
    if (!loadDataset(train, test))
    {
        if (rank == ROOT) std::cerr << "No se pudo cargar MNIST (ejecuta desde build/)" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    test.erase(test.begin() + std::min<std::size_t>(p.nTest, test.size()), test.end());
    std::vector<TSample> tuneSet(train.begin(),
                                 train.begin() + std::min<std::size_t>(train.size(), p.nTune));

    if (p.speculative && p.evalProcs <= 0)
    {
        double f = evalFraction(tuneSet.size(), p.folds, p.maxK, test.size(), train.size());
        p.evalProcs = autoEvalProcs(f, size);
        if (rank == ROOT)
            std::cout << "Fraccion estimada de evaluacion: " << f * 100 << "% -> " << p.evalProcs
                      << " proceso(s) de evaluacion" << std::endl;
    }

    Stats st;
    st.compute = MPI_Wtime() - t0; // La carga es trabajo replicado en todos los procesos.
    Outcome out;
    if (rank == ROOT)
        std::cout << "Modo " << (p.speculative ? "spec" : "data") << " con " << size
                  << " procesos" << (p.speculative ? " (" + std::to_string(p.evalProcs) + " de evaluación)" : "")
                  << ", ajuste con " << tuneSet.size() << " muestras, " << p.folds
                  << " folds, K en [1, " << p.maxK << "], " << test.size() << " muestras de test"
                  << std::endl;

    if (p.speculative) runSpeculative(tuneSet, train, test, p, t0, st, out);
    else runDataParallel(tuneSet, train, test, p, t0, st, out);

    // Tiempo total = el del último proceso en terminar.
    double totalTime = 0.0;
    MPI_Reduce(&st.finish, &totalTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);

    double mine[6] = {static_cast<double>(st.role), st.compute, st.comm, st.finish,
                      static_cast<double>(st.restarts), static_cast<double>(st.wasted)};
    std::vector<double> all(rank == ROOT ? 6 * size : 0);
    MPI_Gather(mine, 6, MPI_DOUBLE, all.data(), 6, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) printReport(p, size, out, totalTime, all);

    MPI_Finalize();
    return 0;
}
