PATH := $(PATH):/usr/local/cuda-12.9/bin

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
OUT_DIR := out
LOAD_DIR := ..
GEN_DIR := ../utils

LOADLEVELS := low medium high
THREADSBLOCK := 32 64 128 256 512 1024

EPOCHS_high := 100
EPOCHS_medium := 100
EPOCHS_low := 100
PARTICLENUM_high := 20000
PARTICLENUM_medium := 5000
PARTICLENUM_low := 100

NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_61,code=sm_61 -O3 -Wno-deprecated-gpu-targets
# DEBUG_FLAGS = -Wall -Wextra -O3 -std=c++17 -DDEBUG -fopenmp -g
LDFLAGS += -I$(SRC_DIR)/include

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/include/*.cuh)
OBJECTS = $(wildcard $(OBJ_DIR)/*.o)
TARGETS = $(foreach level,$(LOADLEVELS),$(foreach threads,$(THREADSBLOCK),$(BIN_DIR)/main_$(level)_$(threads)))
OUTPUT_FILES = $(wildcard $(OUT_DIR)/output*.txt)
ERROR_FILES = $(wildcard $(OUT_DIR)/error*.txt)


.PHONY: all clean run debug

all: $(TARGETS)

$(BIN_DIR)/main_%: $(SOURCES) $(HEADERS)
# Retrieve parameters from target name: main_<level>_<threads>
	$(eval PARAMS_ := $(patsubst $(BIN_DIR)/main_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))

	$(NVCC) $(NVCCFLAGS) -o $@ $(SOURCES) $(LDFLAGS) -DPARTICLE_NUM=$(PARTICLENUM_$(level)) -DDURATION=$(EPOCHS_$(level)) -DTHREADS_PER_BLOCK=$(threads)

debug: NVCCFLAGS := $(DEBUG_FLAGS)
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGETS) $(OUTPUT_FILES) $(ERROR_FILES)

test_%: $(BIN_DIR)/main_%
# Retrieve parameters from target name: main_<level>_<threads>
	$(eval PARAMS_ := $(patsubst test_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	./$< 0 $(OUT_DIR)/time.txt < $(LOAD_DIR)/$(level).txt > $(OUT_DIR)/output_$(level)_$(threads).txt 2> $(OUT_DIR)/error_$(level)_$(threads).txt

test: $(foreach level,$(LOADLEVELS),$(foreach threads,$(THREADSBLOCK),test_$(level)_$(threads)))
	@echo "All tests completed."

generate_%: $(GEN_DIR)/generate_config.py
	python3 $< -e $(EPOCHS_$*) -p $(PARTICLENUM_$*) -o $(LOAD_DIR)/$*.txt