CXX=g++
CFLAGS=-Wall -pedantic -g3
all: Oregon_SNN clean

Oregon_SNN: import_data.o aux_func.o Oregon_SNN.o
	$(CXX) $(CFLAGS) -o $@ $^

aux_func.o: aux_func.cpp aux_func.h
import_data.o: import_data.cpp aux_func.h
Oregon_SNN.o: aux_func.cpp aux_func.h Oregon_SNN.cpp

clean:
	rm -f aux_func.o Oregon_SNN.o import_data.o