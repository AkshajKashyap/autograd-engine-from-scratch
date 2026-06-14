# Autograd Engine From Scratch

A NumPy-based autograd engine and tiny neural network library implementing reverse-mode differentiation, tensor operations, neural network layers, optimizers, gradient checks, and small training experiments from scratch.

## Goal

Build enough of a deep learning framework to understand how backpropagation, tensors, layers, optimizers, and training loops actually work internally.

## Current Status

The scalar reverse-mode autodiff engine is complete, and the core NumPy-backed tensor
autograd engine now supports arithmetic, reductions, matrix multiplication, broadcasting,
and common nonlinear functions. Neural network layers and optimizers remain future milestones.
