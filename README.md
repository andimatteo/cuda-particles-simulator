# particles simulator
## compilazione
1. compilazione con output `make debug`
2. compilazione senza output `make`
3. esecuzione test `make run`

## eseguire l'applicazione
- una volta compilato, lanciare il comando ./main [option] < input.txt > output.txt.
- le possibili option per il momento sono:
    - 0 --> esecuzione sequenziale su CPU
    - 1 --> esecuzione multithread su CPU (le particelle sono divise in batch equi tra i thread)
        - threads: t1, t2, t3, ..., tn
        - particles: p1, p2, p3, ..., pk
        - come sono ripartiti: p1:t1, p2:t1, ..., p(k/n):tn, p((k+1)/n):t2, ...

## todo
- [] --> dimensionare file di carico in base alle dimensioni delle cache e verificare comportamento
- [] --> fare una coda di thread per bilanciare il carico tra thread
- [] --> portare su GPU
