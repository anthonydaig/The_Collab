CXX=g++
CPPFLAGS= -std=c++11 -I/Users/anthonygagliardi/Desktop/The_BroccolLaboration/eigen-eigen-b30b87236a1b/ -Wall -pedantic -g3
all: SdA clean

SdA: SdA.o
	$(CXX) $(CPPFLAGS) -o $@ $^

SdA.o: SdA.cpp
clean:
	rm -f SdA.o
