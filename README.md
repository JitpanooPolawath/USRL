
# Unsupervised Reinforcement learning (USRL)

This project aims to utilize an unsupervised autoencoder to generate a latent space input for an online reinforcement learning model. Hopefully, increasing generalization. 



## Installation

environment requirement.

```bash
  # Python 3.11
  pip install torch gymnasium minari  
```
    
Running the VAE model to train it on breakout, centipede, and demon attack.
```bash
  python VAE.py
```
## Output

```bash
  encoder_model_all.pth
  vae_model_all.pth
  
  # Testing image
  reconstruction_epoch_final_test.png
```

## Acknowledgements

 - [Check out Alexjman - Conv VAE](https://github.com/alexjmanlove/convolutional-variational-autoencoders/tree/main)
 - [Alexander Van de Kleut - VAE](https://avandekleut.github.io/vae/)

