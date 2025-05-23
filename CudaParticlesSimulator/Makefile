NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_61,code=sm_61 -O3
# DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG -fopenmp -g
LDFLAGS += -Iinclude

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOAD_DIR = ..
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/include/*.cuh)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/main
OUTPUT_FILE = out/output.txt

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

debug: NVCCFLAGS := $(DEBUG_FLAGS)
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET) $(OUTPUT_FILE)

test: all
	./$(TARGET) 0 < $(LOAD_DIR)/low.txt > $(OUTPUT_FILE)