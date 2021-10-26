HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

HEADERS = kernel.h
SOURCES = gemm.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=gemm

.PHONY: test


all: $(EXECUTABLE)# test

CXXFLAGS =-O3

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) $(kernel.h)
	$(HIPCC) $(OBJECTS) -o $@

# run test
test: $(EXECUTABLE)
	./$(EXECUTABLE) 256 128 32

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)


