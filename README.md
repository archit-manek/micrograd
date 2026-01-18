# Micrograd (from Karpathy): learning autograd & backprop

This repo is my re-implementation of Andrej Karpathy’s micrograd. I did it to get a concrete feel for how automatic differentiation and backpropagation work under the hood before leaning on PyTorch.

## Why I built this:

I can follow backprop on paper, but I wanted to understand what’s actually happening when gradients flow through a computation. Writing a tiny scalar autograd engine forced me to make the chain rule and gradient accumulation feel less abstract.

## What I learned:

Backpropagation is essentially local nodes communicating with their immediate neighbors.
* **The "Gradient Ahead":** I realized that every node only needs to know two things: its own local derivative and the gradient of the node *ahead* of it (the child node).
* **Global Coordination:** By recursively applying this simple rule via the Chain Rule, the entire graph coordinates to minimize loss.
  
One of the trickiest parts of the implementation was handling variables used multiple times (e.g., when `x` flows into both `y` and `z`).
* **The realization:** If a variable affects the output through multiple paths, its gradients must accumulate (add up), not overwrite each other.
* **Mathematically:** This is the multivariate chain rule in action.
* **In Code:** This is why we do `self.grad += ...` instead of `self.grad = ...` in the backward pass.

## What’s implemented:

* **Autograd Engine:** Implements backward passes for basic operations (`+`, `*`, `tanh`, `exp`).
* **Topological Sort:** To propagate gradients, I implemented a topological sort to ensure we calculate gradients in the correct dependency order (parents before children).
* **Visualization:** I added `graphviz` support to render the computational graph, allowing me to visually inspect the "flow" of data and gradients.