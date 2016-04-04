CXX=g++
CPPFLAGS= -std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: PCA clean

PCA: PCA.o
	$(CXX) $(CPPFLAGS) -o $@ $^

PCA.o: PCA.cpp
clean:
	rm -f PCA.o
