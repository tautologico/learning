all: basic

mlpnnets.o: mlpnnets.c mlpnnets.h
	gcc -Wall -c mlpnnets.c

basictests.o: basictests.c mlpnnets.h
	gcc -Wall -c basictests.c

testiris.o: testiris.c mlpnnets.h
	gcc -Wall -c testiris.c

basic: mlpnnets.o basictests.o
	gcc -o basic mlpnnets.o basictests.o

iris: testiris.o mlpnnets.o
	gcc -o iris testiris.o mlpnnets.o

adult: testadult.o mlpnnets.o
	gcc -o adult testadult.o mlpnnets.o

clean:
	rm -f *.o
	rm -f *.gch
	rm -f basic
	rm -f iris
	rm -f adult
