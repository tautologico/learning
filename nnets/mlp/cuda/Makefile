all: xor_train

testfwd: testfwd.o mlpnnets.o
	nvcc -o testfwd testfwd.o mlpnnets.o -lcurand

testfwd.o: testfwd.cu
	nvcc -c testfwd.cu

xor_train.o: xor_train.cu
	nvcc -c xor_train.cu

xor_train: xor_train.o
	nvcc -o xor_train xor_train.o -lcurand

mlpnnets.o: mlpnnets.cu mlpnnets.h
	nvcc -c mlpnnets.cu -lcurand

clean:
	rm -f *.o
	rm -f xor_train
	rm -f testfwd
