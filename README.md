# Kitware Generated Image Detector

## Overview

This repository contains software to detect computer-generated images, such as those created by Generative Adversarial Network (GAN) algorithms, distinguishing them from typical photos captured by cameras. A computer generated image is a strong indication that the content is fabricated, which is vital to ascertain for digital image forensics. 

The [slides](doc/kitware_SG3_generated.pdf) provide an overview of the approach and the detection results on NVIDIA's [StyleGAN3 image generator](https://github.com/NVlabs/stylegan3). This algorithm was one of the detectors evaluated on StyleGAN3 images, see the details [here](https://github.com/NVlabs/stylegan3-detector#results). Overall, this detector was able to achieve a strong performance of **0.92** Area Under the ROC Curve (AUC) metric (1.0 being the perfect score) on this test data.

For further information please contact Dr. Harry Sun (<harry.sun@kitware.com>).

## License
This software is provided the uses an permissive [BSD License](LICENSE).

## Acknowledgements
This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0123. The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

Distribution A: Approved for public release: distribution unlimited.
