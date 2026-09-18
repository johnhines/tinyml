CC := cc

TARGET := build/neuron
SRC := src/neuron.c

CFLAGS := \
	-std=c17 \
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wshadow \
	-Wconversion


$(TARGET): $(SRC) | build
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) 


