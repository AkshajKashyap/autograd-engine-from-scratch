# Autograd Engine From Scratch

A NumPy-based autograd engine and tiny neural network library implementing reverse-mode differentiation, tensor operations, neural network layers, optimizers, gradient checks, and small training experiments from scratch.

## Goal

Build enough of a deep learning framework to understand how backpropagation, tensors, layers, optimizers, and training loops actually work internally.

## Current Status

The first milestone is complete: a scalar reverse-mode autodiff engine with arithmetic
operations, common nonlinear functions, topological backpropagation, and gradient checks.
