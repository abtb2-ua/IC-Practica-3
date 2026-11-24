#!/usr/bin/env bash
# Batería de pruebas de la tarea 4 (paso e). Ejecutar desde knn2/:  ./bench.sh
# Cada ejecución añade una fila a build/results/results.csv y guarda su salida completa en
# build/results/logs/. Va de menor a mayor tamaño: si se interrumpe, lo hecho sigue sirviendo.
#
# Variables opcionales:
#   PROCS="2 4 6 8"                  número de procesos MPI a probar
#   MPI_OPTS="--use-hwthread-cpus"   opciones extra de mpirun (p. ej. "-hostfile hosts" en el clúster)
#   SIZES="2000:500 5000:2000 10000:5000"   pares nTune:nTest
set -u
PROCS=${PROCS:-"2 4 6 8"}
MPI_OPTS=${MPI_OPTS:-"--use-hwthread-cpus"}
SIZES=${SIZES:-"2000:500 5000:2000 10000:5000"}
SWEEP=${SWEEP:-1}   # 0 = no repetir el barrido de evalProcs
FOLDS=5
MAXK=10

cd "$(dirname "$0")/build" || exit 1
OUT=results
mkdir -p "$OUT/logs"
# Solo una batería a la vez: dos en paralelo se reparten los núcleos y los tiempos no valen.
exec 9> "$OUT/.lock"
if ! flock -n 9; then echo "Ya hay un bench.sh en marcha. Páralo antes: pkill -f bench.sh" >&2; exit 1; fi
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "mode,procs,evalProcs,nTune,folds,maxK,nTest,order,bestK,accuracy,total,tune,eval,compute,comm,wasted" > "$CSV"

# Caracterización de la máquina para la memoria.
{ lscpu; echo; free -h; echo; command -v lstopo-no-graphics >/dev/null && lstopo-no-graphics --no-io; } \
    > "$OUT/arquitectura.txt" 2>&1

run() { # run <nombre_log> <comando...>
    local log="$OUT/logs/$1.txt"; shift
    echo "[$(date +%H:%M:%S)] $*"
    "$@" > "$log" 2>&1
    local line; line=$(grep '^RESULT,' "$log" | sed 's/^RESULT,//')
    if [ -n "$line" ]; then echo "$line" >> "$CSV"; echo "    -> $line"
    else echo "    !! sin línea RESULT, revisa $log"; fi
}

for size in $SIZES; do
    nTune=${size%%:*}; nTest=${size##*:}
    tag="${nTune}_${nTest}"
    run "seq_${tag}" ./cv "$nTune" $FOLDS $MAXK "$nTest"
    for p in $PROCS; do
        run "data_p${p}_${tag}" mpirun $MPI_OPTS -np "$p" ./cv_mpi "$nTune" $FOLDS $MAXK "$nTest" data
        run "spec_asc_p${p}_${tag}" mpirun $MPI_OPTS -np "$p" ./cv_mpi "$nTune" $FOLDS $MAXK "$nTest" spec 0 asc
        run "spec_desc_p${p}_${tag}" mpirun $MPI_OPTS -np "$p" ./cv_mpi "$nTune" $FOLDS $MAXK "$nTest" spec 0 desc
    done
done

# Barrido del reparto ajuste/evaluación en spec (pregunta 10), tamaño intermedio y P máximo.
pmax=${PROCS##* }
[ "$SWEEP" = 1 ] && for e in 1 2 3 4; do
    [ "$e" -lt "$pmax" ] || continue
    run "spec_asc_p${pmax}_e${e}_5000_2000" mpirun $MPI_OPTS -np "$pmax" ./cv_mpi 5000 $FOLDS $MAXK 2000 spec "$e" asc
done

echo "Terminado. Resultados en build/$CSV"
