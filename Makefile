HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES = gemm.cpp kernel.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=gemm

.PHONY: test


all: $(EXECUTABLE)# test

CXXFLAGS =-O3

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) $(HEADERS)
	$(HIPCC) $(OBJECTS) -o $@

# run test
test: $(EXECUTABLE)
	./$(EXECUTABLE) 256 128 32

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)


