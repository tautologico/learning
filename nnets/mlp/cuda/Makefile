all: xor_train

xor_train.o: xor_train.cu
	nvcc -c xor_train.cu

xor_train: xor_train.o
	nvcc -o xor_train xor_train.o -lcurand

mlpnnets.o: mlpnnets.cu mlpnnets.h
	nvcc -c mlpnnets.cu -lcurand

clean:
	rm -f *.o
	rm -f xor_train
