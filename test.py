import jax
import jax.numpy as jnp

print(jax.devices())

x = jnp.array([1.0, 2.0, 3.0])
print(jax.device_get(x))