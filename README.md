# CANARY

## files

`/canary/` a symlink to `/src/` for the sake of pythons packaging.
`/minetest/` a symlink to the minetest project directory, currently unused.
`/jnb/` a directory of .ipynb files showcasing the capabilities of this project.
`/res/` resource files read by notebooks and source files
`/res/prm/` read-only model parameters
`/run/` data generated from running the model
`/run/$run_id/metadata.json` instructions on how to read the data from this run.

## todo

gather data

- obs
- reward
- info
- activations
  - conv
  - deep
  - actor
  - critic

interpret

1. find a single neuron that activates when wood is pointed at
1. probe activations

## ideas

apply PCA

interpretable models?

- incentivise sparse activations
- randomly sample weights

Better dropout with [path patching](https://arxiv.org/abs/2304.05969).
Instead of setting activations to zero, set them to activations of a different data point.

Causal scrubbing and path patching do resampling ablation.
They acknowledge that the model does not work well OOD.
This implies we can quantify OOD-ness, guess:

- How much Guassian noise can you add before path patching stops working?
  more questions:
- Which features are more sensitive to guassian noise wrt OOD-ness?
- Which features often co-occur (or, what is a feature)?
