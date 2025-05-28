PATH := $(PATH):/usr/local/cuda-12.9/bin

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
OUT_DIR := out
LOAD_DIR := ..
GEN_DIR := ../utils

LOADLEVELS := low medium high higher
THREADSBLOCK := 32 64 128 256 512 1024
VERSIONS := 0 1 2 3 4 5

EPOCHS := 4
PARTICLENUM_higher := 60000
PARTICLENUM_high := 20000
PARTICLENUM_medium := 5000
PARTICLENUM_low := 100

COMPUTE_CAPABILITY := $(shell if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g'; else echo 75; fi)
NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -O3 -Wno-deprecated-gpu-targets -lineinfo
EXTRA_NVCCFLAGS = $(if $(filter 4 5,$(version)),-use_fast_math,)
LDFLAGS += -I$(SRC_DIR)/include

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/include/*.cuh)
OBJECTS = $(wildcard $(OBJ_DIR)/*.o)
TARGETS := $(foreach level,$(LOADLEVELS), \
			$(foreach threads,$(THREADSBLOCK), \
				$(foreach version,$(VERSIONS), \
						$(BIN_DIR)/main_$(level)_$(threads)_$(version))))
TESTS 	= $(foreach level,$(LOADLEVELS), \
			$(foreach threads,$(THREADSBLOCK), \
				$(foreach version,$(VERSIONS), \
						test_$(level)_$(threads)_$(version))))
OUTPUT_FILES = $(wildcard $(OUT_DIR)/output*.txt)
ERROR_FILES = $(wildcard $(OUT_DIR)/error*.txt)
TIME_FILES = $(wildcard $(OUT_DIR)/time*.txt)

NSIGHT := /usr/local/NVIDIA-Nsight-Compute-2025.2/target/linux-desktop-glibc_2_11_3-x64/ncu
REPORT = "/home/taulant/Documents/NVIDIA Nsight Compute/cuda-particles-simulator/report_$(level)_$(threads)_$(version)_%i"

.PHONY: all clean test

all: $(TARGETS)

$(BIN_DIR)/main_%: $(SOURCES) $(HEADERS)
# Retrieve parameters from target name: main_<level>_<threads>_<version>
	$(eval PARAMS_ := $(patsubst $(BIN_DIR)/main_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	$(eval version := $(word 3,$(PARAMS)))

	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -o $@ $(SOURCES) $(LDFLAGS) -DPARTICLE_NUM=$(PARTICLENUM_$(level)) -DDURATION=$(EPOCHS) -DTHREADS_PER_BLOCK=$(threads) -DVERSION=$(version)

clean:
	rm -f $(OBJECTS) $(TARGETS) $(OUTPUT_FILES) $(ERROR_FILES) $(TIME_FILES) bin/main_*

test_%: $(BIN_DIR)/main_%
# Retrieve parameters from target name: test_<level>_<threads>_<version>
	$(eval PARAMS_ := $(patsubst test_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	$(eval version := $(word 3,$(PARAMS)))

	./$< $(OUT_DIR)/time.txt < $(LOAD_DIR)/$(level).txt > $(OUT_DIR)/output_$(level)_$(threads)_$(version).txt 2> $(OUT_DIR)/error_$(level)_$(threads)_$(version).txt

test: $(TESTS)
	@echo "All tests completed."

nsight_%: $(BIN_DIR)/main_%
# Retrieve parameters from target name: nsight_<level>_<threads>_<version>
	$(eval PARAMS_ := $(patsubst nsight_%,%,$@))
	$(eval PARAMS = $(subst _, ,$(PARAMS_)))
	$(eval level := $(word 1,$(PARAMS)))
	$(eval threads := $(word 2,$(PARAMS)))
	$(eval version := $(word 3,$(PARAMS)))

	sudo $(NSIGHT) --config-file off --export $(REPORT) --force-overwrite --set full $< $(OUT_DIR)/time.txt < $(LOAD_DIR)/$(level).txt

# low medium high
generate_%: $(GEN_DIR)/generate_config.py
	python3 $< -e $(EPOCHS) -p $(PARTICLENUM_$*) -o $(LOAD_DIR)/$*.txt