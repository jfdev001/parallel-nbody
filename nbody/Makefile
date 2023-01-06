CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2

all: nbody-seq nbody-par

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm

nbody-par: nbody-par.c
	mpicc -o nbody-par nbody-par.c -lm

clean:
	rm -f *.o nbody-seq *~ *core nbody-par
