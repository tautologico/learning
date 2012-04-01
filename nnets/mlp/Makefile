all: basic

mlpnnets.o: mlpnnets.c mlpnnets.h
	gcc -Wall -c mlpnnets.c

basictests.o: basictests.c mlpnnets.h
	gcc -Wall -c basictests.c

basic: mlpnnets.o basictests.o
	gcc -o basic mlpnnets.o basictests.o

clean:
	rm -f *.o
	rm -f basic
