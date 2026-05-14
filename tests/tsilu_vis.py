import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Input range
g = torch.linspace(-8, 8, 2000)

# Temperature values, including tau = 4
taus = [0.5, 1.0, 2.0, 4.0]

plt.figure(figsize=(7, 5))

for tau in taus:
    # Magnitude-preserving temperature-scaled SiLU
    y = tau * F.silu(g / tau)
    plt.plot(g.numpy(), y.numpy(), label=fr"$\tau={tau}$")

plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel(r"Gate preactivation $g$")
plt.ylabel(r"$\tau \cdot \mathrm{SiLU}(g / \tau)$")
plt.title("Magnitude-preserving temperature-scaled SiLU")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
