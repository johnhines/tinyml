CC := cc

TARGET := build/neuron build/vector
SRC := examples/neuron.c examples/vector.c

CFLAGS := \
	-std=c17 \
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wshadow \
	-Wconversion


all: build/neuron build/vector

build/%:	examples/%.c 
	$(CC) $(CFLAGS) $< -o $@

# $(TARGET): $(SRC) | build
# 	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 


