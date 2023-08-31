import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Callable


class Network(nn.Module):
    @nn.compact
    def __call__(self, x, start_at=0):
        if start_at < 1:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x / (255.0)
            x = nn.Conv(
                32,
                kernel_size=(8, 8),
                strides=(4, 4),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv0", x)

        if start_at < 2:
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv1", x)

        if start_at < 3:
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv2", x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)


def load_params(
    key,
    network,
    actor,
    critic,
    saved_model_path,
    action_dim=36,
    obs_shape=(1, 4, 64, 64),
    load_perturbations=True,
):
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    sample_obs = np.zeros(obs_shape, dtype=np.float32)
    print(sample_obs.shape)

    network_params = network.init(network_key, sample_obs)
    actor_params = actor.init(actor_key, network.apply(network_params, sample_obs)[0])
    critic_params = critic.init(
        critic_key, network.apply(network_params, sample_obs)[0]
    )

    if load_perturbations:
        perturbations = network_params.pop("perturbations")

    with open(saved_model_path, "rb") as f:
        (
            _,
            (network_params, actor_params, critic_params),
        ) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    if load_perturbations:
        return key, network_params, actor_params, critic_params, perturbations
    else:
        return key, network_params, actor_params, critic_params
