all: xor_train

xor_train.o: xor_train.cu
	nvcc -c xor_train.cu

xor_train: xor_train.o
	nvcc -o xor_train xor_train.o -lcurand

clean:
	rm -f *.o
	rm -f xor_train
