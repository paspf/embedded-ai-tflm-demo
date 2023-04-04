/**
  ******************************************************************************
  * @file           : test_gates.c
  * @brief          : Test neural networks, trained to represent logical gates.
  * @author			: paspf
  * @date			: 2023-04-01
  * @copyright		: paspf, GNU General Public License v3.0
  ******************************************************************************
  */

#ifndef TEST_GATES_G
#define TEST_GATES_G

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#include <stdint.h>
// Let test_gates look like a c function.
EXTERNC void test_gates();

#undef EXTERNC

int test_and_gate();
int test_nor_gate();
int test_or_gate();
int test_xnor_gate();
int test_xor_gate();

int8_t float_to_quant(float input, float scale, int zero_point);
float quant_to_float(int32_t quant, float scale, int zero_point);
int float_prediction_to_binary_int(const float prediction);
int print_single_prediction(const float *x_input, const int y_ref, const float y_pred_float, const int y_pred_int);
int predict(void (*model_inference)(float*, float *), const int* output_data_reference);

#endif // TEST_GATES_G
