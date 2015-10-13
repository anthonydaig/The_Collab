CXX=g++
CFLAGS=-Wall -pedantic -g3
all: SNN clean

SNN: snn_func.o SNN.o
	$(CXX) $(CFLAGS) -o $@ $^

snn_func.o: snn_func.cpp snn_func.h
SNN.o: snn_func.cpp snn_func.h SNN.cpp

clean:
	rm -f snn_func.o SNN.o