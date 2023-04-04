# AI Gates using TensorFlow Lite for Microcontrollers (TFLM)

This project runs small neural networks in [TFLM](https://github.com/tensorflow/tflite-micro). The networks are trained to predict logical gates (AND, OR, NOR, XOR, XNOR).

The networks were trained with Tensorflow `2.10` and brought to the microcontroller (STM32 L476RG) using CubeIDE `1.11.2`. This git repo includes the associated Cube IDE project, the Jupyter Notebook required to train the neural networks, as well as a Jupyter Notebook to convert the neural networks for TFLM.

The demo application includes the following features:
- Predict functions for all gates (test_gates.cpp)
- Print model evaluation to command line interface (test_gates.cpp)
- Map a single gate to I/O ports (io_gate.cpp)
- Usage of the [ES-PCB for Nucleo L476RG](https://github.com/paspf/ES-PCB-for-L476RG) as periphery board

## Setup

### Get Repository

```
git clone git@github.com:paspf/embedded-ai-tflm-demo.git
```

This repository already includes TFLM and all required third party tools.

### Open Project in Cube IDE

Open the repositories folder in Cube IDE as workspace, import the `tflm-gates-cube-project`.

## Program Output

When running the program, the command line output should look similar to this:
```
test_xor_gate_tflm_8bit<\n>
|[0.0, 0.0]|[0]| -> 0.000000 -> Pass<\n>
|[0.0, 1.0]|[1]| -> 0.996094 -> Pass<\n>
|[1.0, 0.0]|[1]| -> 0.996094 -> Pass<\n>
|[1.0, 1.0]|[0]| -> 0.000000 -> Pass<\n>

```
