CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2

all: nbody-seq nbody-par nbody-seq-test

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm

nbody-par: nbody-par.c
	mpicc $(CFLAGS) -o nbody-par nbody-par.c -lm
	# mpicc -o nbody-par nbody-par.c -lm

nbody-seq-test: nbody-seq-test.c 
	gcc -o nbody-seq-test nbody-seq-test.c -lm -g

clean:
	rm -f *.o nbody-seq *~ *core nbody-par nbody-seq-test
