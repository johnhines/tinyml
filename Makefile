CC := cc

CPPFLAGS := -Iinclude

CFLAGS := \
	-std=c17 \
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wshadow \
	-Wconversion \
	-O0 \
	-g3

BUILD_DIR := build

.PHONY: all clean \
	01_neuron 02_matrix 03_dense \
	run-01_neuron run-02_matrix run-03_dense

all: 01_neuron 02_matrix 03_dense


# ------------------------------------------------------------
# Example targets
# ------------------------------------------------------------

01_neuron: $(BUILD_DIR)/01_neuron

02_matrix: $(BUILD_DIR)/02_matrix

03_dense: $(BUILD_DIR)/03_dense


# ------------------------------------------------------------
# Executables
# ------------------------------------------------------------

$(BUILD_DIR)/01_neuron: examples/01_neuron.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

$(BUILD_DIR)/02_matrix: examples/02_matrix.c src/matrix.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/03_dense: examples/03_dense.c src/matrix.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@


# ------------------------------------------------------------
# Run targets
# ------------------------------------------------------------

run-01_neuron: 01_neuron
	./$(BUILD_DIR)/01_neuron

run-02_matrix: 02_matrix
	./$(BUILD_DIR)/02_matrix

run-03_dense: 03_dense
	./$(BUILD_DIR)/03_dense


# ------------------------------------------------------------
# Build directory / cleanup
# ------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
