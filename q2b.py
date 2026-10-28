# Carnegie Mellon University 16-280 Machine Learning HW F25
# Created by Julius Arolovitch, jarolovi@andrew.cmu.edu
# Last modified July 2025

import math
import torch

import matplotlib
matplotlib.use("Agg") # This setting is necessary for running on remote machines
import matplotlib.pyplot as plt

############ START STUDENT CODE ############

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

learning_rate = 1e-6 # Do not change

# Run 2000 iterations of gradient descent
for t in range(2000):
    # inside the loop
    y_pred = a * x ** 3 + b * x ** 2 + c * x + d
 
    # Hint: (3-5).pow(2) = (3-5) * (3-5) = 4. .sum() still holds from Q2 :)
    loss = (y_pred - y).pow(2).sum()


    if t % 100 == 99:
        print(f"Iteration {t} loss: {loss}") # Print loss every 100 iterations

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} x^3 + {b.item()} x^2 + {c.item()} x + {d.item()}')

############ END STUDENT CODE ############

# Plotting domain (tensor)
x_plot = torch.linspace(-2 * math.pi, 2 * math.pi, 4000)
y_plot = torch.sin(x_plot)
y_pred_plot = a * x_plot ** 3 + b * x_plot ** 2 + c * x_plot + d

plt.figure(figsize=(10, 4))
plt.plot(x_plot.numpy(), y_plot.numpy(), label='sin(x)')
plt.plot(x_plot.numpy(), y_pred_plot.detach().numpy(), label='Learned polynomial')
plt.title('Q3b')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.ylim(-2, 2)

plt.tight_layout()

plt.savefig('output_3b.png')
plt.close()
