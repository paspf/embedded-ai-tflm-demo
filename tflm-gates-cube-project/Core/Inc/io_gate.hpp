/**
 ******************************************************************************
 * @file            : io_gate.h
 * @brief           : Map a gate to gpio ports.
 * @author			: paspf
 * @date			: 2023-04-03
 * @copyright		: paspf, GNU General Public License v3.0
 ******************************************************************************
 */

#ifndef INC_IO_GATE_HPP_
#define INC_IO_GATE_HPP_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void io_gates();

#undef EXTERNC

#endif /* INC_IO_GATE_HPP_ */
