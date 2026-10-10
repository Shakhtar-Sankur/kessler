CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall -Wextra -fopenmp -ffp-contract=off
LIB = kessler/_libkessler.so

all: $(LIB)

$(LIB): src/capi.cpp include/kessler/sgp4.hpp
	$(CXX) $(CXXFLAGS) -Iinclude -shared src/capi.cpp -o $@

clean:
	rm -f $(LIB)

.PHONY: all clean
