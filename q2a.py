import torch

# Read docs here: https://pytorch.org/docs/stable/generated/torch.tensor.html
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x ** 2 + y ** 2

# For clarity, this is not a loss function, it's just a value we're computing.
loss = z ** 2

# Print the loss value
print(f"Loss: {loss.item()}")

# Gradients should be: dloss/dx = 4x * z, dloss/dy = 4y * z
loss.backward()

# w.r.t. means with respect to - will come in handy in the future :)
print(f"Gradient of loss w.r.t x: {x.grad}")
print(f"Gradient of loss w.r.t y: {y.grad}")
