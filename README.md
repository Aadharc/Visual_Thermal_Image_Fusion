# Image Fusion for Misaligned Visual and Thermal Images

This repository contains Python scripts and resources for image fusion of misaligned thermal and visual images. The goal is to create a fusion network that can effectively combine information from thermal and visual images, even when they are not perfectly aligned.

## Table of Contents

- [Dataset2.py](#dataset2py)
- [Discriminator.py](#discriminatorpy)
- [GenAttn.py](#genattnpy)
- [loss.py](#losspy)
- [main.py](#mainpy)
- [Metrics.py](#metricspy)
- [utils.py](#utilspy)

## Scripts

### Dataset2.py

- **Description**: This script is used to load the WiSARD dataset after post-processing to improve fusion results.
- **Usage**: You can use this script to load and preprocess the dataset before training your fusion network.

### Discriminator.py

- **Description**: This Python file contains the code for the discriminator in the GAN (Generative Adversarial Network) architecture.
- **Usage**: The discriminator is a crucial component of GANs and is responsible for distinguishing between real and generated images. You can integrate this script into your GAN-based fusion network.

### GenAttn.py

- **Description**: This script implements the generator for the GAN (Generative Adversarial Network) with an attention mechanism. The attention mechanism helps in fusing misaligned images by focusing on important features from each modality while discarding unnecessary ones.
- **Usage**: Use this script to create the generator of your fusion network, which incorporates attention mechanisms for improved fusion results.

### loss.py

- **Description**: This Python file contains the code for a customized loss function tailored for the image fusion task.
- **Usage**: Incorporate this customized loss function into your training process to optimize your fusion network for better results.

### main.py

- **Description**: This script is the main entry point for training your fusion network. It includes the training loop, model initialization, and other necessary steps for training your network.
- **Usage**: Execute this script to start training your fusion network.

### Metrics.py

- **Description**: This script provides functions for the evaluation of your fusion network method. It includes metrics for assessing the quality of the fused images.
- **Usage**: Utilize this script to evaluate the performance of your fusion network after training.

### utils.py

- **Description**: This file contains various utility functions required for training and working with your fusion network.
- **Usage**: Include this script in your project to access utility functions that streamline the training process.

## Getting Started

To get started with this repository, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Aadharc/Visual_Thermal_Image_Fusion.git
   ```

2. Install the necessary dependencies if not already installed:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the scripts mentioned above for your specific use case and customize them as needed.

4. Run `main.py` to start training your fusion network:

   ```bash
   python main.py
   ```

5. Use `Metrics.py` to evaluate the performance of your trained fusion network.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the [WiSARD](https://sites.google.com/uw.edu/wisard/home) dataset for providing valuable data for this project.
## Contact

For any further information or inquiries, feel free to contact the project maintainers:

- [Aadhar Chauhan](mailto:aadharc@uw.edu)

Thank you for using this image fusion repository!
