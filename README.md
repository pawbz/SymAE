# SymAE

The repo provides notebooks to reproduce examples in the paper: 
**Redatuming physical systems using symmetric autoencoders**
Pawan Bharadwaj, Matt Li, Laurent Demanet

```
@article{bharadwaj2021redatuming,
  title={Redatuming physical systems using symmetric autoencoders},
  author={Bharadwaj, Pawan and Li, Matthew and Demanet, Laurent},
  journal={arXiv preprint arXiv:2108.02537},
  year={2021}
}
```

Abstract:

This paper considers physical systems described by hidden states and indirectly observed through repeated measurements corrupted by unmodeled nuisance parameters. 
A network-based representation learns to disentangle the coherent information (relative to the state) from the incoherent nuisance information (relative to the sensing). 
Instead of physical models, the representation uses symmetry and stochastic regularization to inform an autoencoder architecture called SymAE.
It enables redatuming, i.e., creating virtual data instances where the nuisances are uniformized across measurements.
