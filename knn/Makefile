CXX=g++
CFLAGS=-Wall -pedantic -g3
all: knn_better clean

knn_better: knn_better.o minheap.o
	$(CXX) $(CFLAGS) -o $@ $^

minheap.o: minheap.cpp minheap.h
knn_better.o: minheap.cpp minheap.h knn_better.cpp

clean:
	rm -f minheap.o knn_better.o