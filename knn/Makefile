CXX=g++
CPPFLAGS=-std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: knn_better clean

knn_better: knn_better.o minheap.o
	$(CXX) $(CPPFLAGS) -o $@ $^

minheap.o: minheap.cpp minheap.h
knn_better.o: minheap.cpp minheap.h knn_better.cpp

clean:
	rm -f minheap.o knn_better.o