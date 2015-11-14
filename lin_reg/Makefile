CXX=g++
CPPFLAGS= -std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: lin_reg clean

lin_reg: lin_reg.o
	$(CXX) $(CPPFLAGS) -o $@ $^

knn_better.o: lin_reg.cpp
clean:
	rm -f lin_reg.o