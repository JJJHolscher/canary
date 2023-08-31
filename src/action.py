import jax.numpy as jnp

# action/id mapping
# Punching is enabled by default
# mouse values are (x,y) pairs
MAP = {
    0: "mouse -25 -25",
    1: "mouse -25 0",
    2: "mouse -25 25",
    3: "mouse 0 -25",
    4: "mouse 0 0",  # null op
    5: "mouse 0 25",
    6: "mouse 25 -25",
    7: "mouse 25 0",
    8: "mouse 25 25",
    9: "mouse -25 -25",
    10: "mouse -25 0, JUMP",
    11: "mouse -25 25, JUMP",
    12: "mouse 0 -25, JUMP",
    13: "mouse 0 0, JUMP",
    14: "mouse 0 25, JUMP",
    15: "mouse 25 -25, JUMP",
    16: "mouse 25 0, JUMP",
    17: "mouse 25 25, JUMP",
    18: "mouse -25 -25, JUMP",
    19: "mouse -25 0, FORWARD",
    20: "mouse -25 25, FORWARD",
    21: "mouse 0 -25, FORwARD",
    22: "mouse 0 0, FORWARD",
    23: "mouse 0 25, FORWARD",
    24: "mouse 25 -25, FORWARD",
    25: "mouse 25 0, FORWARD",
    26: "mouse 25 25, FORWARD",
    27: "mouse -25 -25, FORWARD, JUMP",
    28: "mouse -25 0, FORWARD, JUMP",
    29: "mouse -25 25, FORWARD, JUMP",
    30: "mouse 0 -25, FORWARD, JUMP",
    31: "mouse 0 0, FORWARD, JUMP",
    32: "mouse 0 25, FORWARD, JUMP",
    33: "mouse 25 -25 FORWARD, JUMP",
    34: "mouse 25 0, FORWARD, JUMP",
    35: "mouse 25 25, FOWARD, JUMP",
}
FORWARD_MASK = jnp.array([0] * 18 + [1] * 18)
JUMP_MASK = jnp.array(([0] * 9 + [1] * 9) * 2)
UP_MASK = jnp.array([1, 0, 0] * 12)
DOWN_MASK = jnp.array([0, 0, 1] * 12)
LEFT_MASK = jnp.array(([1] * 3 + [0] * 6) * 4)
RIGHT_MASK = jnp.array(([0] * 6 + [1] * 3) * 4)
