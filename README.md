# generative-vae
Building and training a generative model on MNIST digit data by creating a variational autoencoder. From the introduction in the autoencoder notebook:

The variational autoencoder is an unsupervised method to find embeddings in a lower dimensional space, using an Encoder and a Decoder. The Encoder maps input data in some space $X$ into an embedding space $Z$, typically of lower dimension than $X$, and the Decoder maps encoded (embedded) data from $Z$ back into $X$ space, also denoted $O$ for output space.

Let $\phi$ represent the parameters of the Encoder and $\theta$ the parameters of the Decoder. Then we want to train our autoencoder by minimizing the distance between our input data $X$ and the output data produced by encoding then decoding the input data:
$$min_{\theta, \phi}\sum_{i=1}^N d(X_i, Dec(En(X,\phi),\theta))$$

We'll use logistic loss, so our distance metric becomes the "binary cross-entropy" loss:
$$d(x, Dec(En(X,\phi), \theta))=-\sum_{s}x_s\textrm{ log }Dec(En(x,\phi), \theta)_s-(1-x_s)\textrm{ log }[1-Dec(En(x,\phi), \theta)_s]$$

For the variational autoencoder, our embedding space isn't an explicit "encoding" of the input data $X$, but rather an embedding of the mean and variance of the input. When we decode this embedding, we take the mean $\mu(X)$ and variance $\sigma(X)$ and sample some $Z=\mu(X)+\epsilon\cdot\sigma(X)$ where $\epsilon\sim N(0,I)$ is some noise.

Our loss therefore has a second component based on this "embedding" distribution, which is:
$$-\frac{1}{2}\sum_{j=1}^d 1+\textrm{log }\sigma_j^2(X_i)-\mu_j(X_i)^2-\sigma_j^2(X_i)$$

Once we have an Encoder and a Decoder trained, we can use the Decoder as a generative model by sampling from a standard normal and attempting to "decode" this distribution.
