/**
 ******************************************************************************
 * @file            : io_gate.c
 * @brief           : Map a gate to gpio ports.
 * @author			: paspf
 * @date			: 2023-04-03
 * @copyright		: paspf, GNU General Public License v3.0
 ******************************************************************************
 */

#include <stdio.h>
#include "io_gate.hpp"
#include "stm32l4xx_hal.h"
#include "test_gates.hpp"

/* Include models */
extern unsigned int model_and_gate_len;
extern unsigned char model_and_gate[];

/* TFLM includes */
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Private variables */
static float input_data[2];
static float output_data[1];

/* Private function prototypes */
static void _gpio_init();
static void _gpio_read();
static void _nn_predict();
static void _gpio_write();

/* Register all operations required for the networks */
namespace {
using GatesOpsResolver = tflite::MicroMutableOpResolver<2>;

TfLiteStatus RegisterOps(GatesOpsResolver &op_resolver) {
	TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
	TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
	return kTfLiteOk;
}
}  // namespace.

/**
 * @fn void io_gates()
 * @brief IO gates initialization and main loop.
 *
 */
void io_gates() {
	_gpio_init();
	while(1) {
		_gpio_read();
		_nn_predict();
		_gpio_write();
		HAL_Delay(50);
	}
}

/**
 * @fn void _gpio_init()
 * @brief Initialize GPIO pins.
 *
 */
static void _gpio_init() {
	/* Buttons */
	__HAL_RCC_GPIOB_CLK_ENABLE();
	GPIO_InitTypeDef GPIO_Buttons = {0};
	GPIO_Buttons.Mode = GPIO_MODE_INPUT;
	GPIO_Buttons.Pull = GPIO_PULLDOWN;
	GPIO_Buttons.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_Buttons.Pin = GPIO_PIN_8 | GPIO_PIN_9;

	HAL_GPIO_Init(GPIOB, &GPIO_Buttons);

	/* LEDS */
	__HAL_RCC_GPIOC_CLK_ENABLE();
	GPIO_InitTypeDef GPIO_Leds;
	GPIO_Leds.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_Leds.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_Leds.Pin = GPIO_PIN_5;
	HAL_GPIO_Init(GPIOC, &GPIO_Leds);
}

/**
 * @fn void _gpio_read()
 * @brief Read out GPIO pins and wirte level into input_data.
 *
 */
static void _gpio_read() {
	int pin8 = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_8);
	int pin9 = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9);
	// Write GPIO level to ai_input buffer.
	input_data[0] = (float) pin8;
	input_data[1] = (float) pin9;
}

/**
 * @fn void __nn_predict()
 * @brief Run prediction on the current values in input_data.
 *
 */
static void _nn_predict() {
	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	const tflite::Model *model = ::tflite::GetModel(model_and_gate);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf("Model provided is schema version %d not equal "
				"to supported version %d.\n", model->version(),
				TFLITE_SCHEMA_VERSION);
	}

	GatesOpsResolver op_resolver;
	RegisterOps(op_resolver);

	// Arena size just a round number. The exact arena usage can be determined
	// using the RecordingMicroInterpreter.
	constexpr int kTensorArenaSize = 2056;
	uint8_t tensor_arena[kTensorArenaSize];

	// Build an interpreter to run the model with
	tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
			kTensorArenaSize);

	// Allocate memory from the tensor_arena for the model's tensors
	if (interpreter.AllocateTensors() != kTfLiteOk) {
		MicroPrintf("Allocate tensor failed.");
		return;
	}

	// Obtain a pointer to the model's input tensor.
	TfLiteTensor *input = interpreter.input(0);

	// Make sure the input has the properties we expect.
	if (input == nullptr) {
		MicroPrintf("Input tensor is null.");
		return;
	}
	// Get the input quantization parameters.
	float input_scale = input->params.scale;
	int input_zero_point = input->params.zero_point;

	// Obtain a pointer to the output tensor.
	TfLiteTensor *output = interpreter.output(0);

	// Get the output quantization parameters.
	float output_scale = output->params.scale;
	int output_zero_point = output->params.zero_point;

	// Fill inputs of neural network.
	input->data.int8[0] = float_to_quant(input_data[0], input_scale,
			input_zero_point);
	input->data.int8[1] = float_to_quant(input_data[1], input_scale,
			input_zero_point);
	// Run prediction.
	interpreter.Invoke();
	// Get outputs of neural network.
	output_data[0] = quant_to_float(output->data.int8[0], output_scale,
			output_zero_point);
}

/**
 * @fn void _gpio_write()
 * @brief Write Hi or Lo to a GPIO bin based on the current value in output_data.
 *
 */
static void _gpio_write() {
	// Read output (predicted y) of neural network.
	int y_pred = float_prediction_to_binary_int(output_data[0]);
	if (y_pred == 1) {
		HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, GPIO_PIN_SET);
	}
	else {
		HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, GPIO_PIN_RESET);
	}

}

