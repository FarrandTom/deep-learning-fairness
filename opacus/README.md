## Local Opacus Example
Self-contained Opacus example- does not require you to install the Opacus library as all of the tooling is contained here. This means we can tinker around with the algorithms and adapt to our needs. 

To run the MNIST example simply: `python mnist.py`

With the default settings this took ~10 minutes (45 seconds per epoch) to run on a 2016 Macbook Air. 

## Developer Note:
The implementation of gradient clipping in autograd_grad_sample.py uses backward hooks to capture per-sample gradients.
The `register_backward hook` function has a known issue being tracked at https://github.com/pytorch/pytorch/issues/598. However, this is the only known way of implementing this as of now (your suggestions and contributions are very welcome). The behaviour has been verified to be correct for the layers currently supported by opacus.
