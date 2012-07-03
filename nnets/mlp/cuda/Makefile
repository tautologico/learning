all: iris

adult: adult.o mlpnnets.o
	nvcc -o adult adult.o mlpnnets.o -lcurand

adult.o: adult.cu mlpnnets.h
	nvcc -c adult.cu

iris: iris.o mlpnnets.o
	nvcc -o iris iris.o mlpnnets.o -lcurand

iris.o: iris.cu mlpnnets.h
	nvcc -c iris.cu

xor_mlp: xor_mlp.o mlpnnets.o
	nvcc -o xor_mlp xor_mlp.o mlpnnets.o -lcurand

xor_mlp.o: xor_mlp.cu mlpnnets.h
	nvcc -c xor_mlp.cu

testfwd: testfwd.o mlpnnets.o
	nvcc -o testfwd testfwd.o mlpnnets.o -lcurand

testfwd.o: testfwd.cu mlpnnets.h
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
	rm -f xor_mlp
	rm -f iris
