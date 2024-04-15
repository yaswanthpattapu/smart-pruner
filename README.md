# Smart-Pruner

## Overview
Smart-Pruner is a Pytorch based module designed to apply various pruning techniques across multiple model-dataset architectures. It simplifies the evaluation process by providing insights into accuracy, energy consumption, and speedup for different pruning techniques.

## Features
- Apply multiple pruning techniques to diverse model-dataset configurations.
- Obtain insights on accuracy, energy consumption, and speedup.
- Easy integration of new pruning methods.

## Installation
1. Clone the Smart-Pruner repository:
   ```shell
   git clone https://github.com/prabhas2002/Smart-Pruner.git
   ```
2 Install the required dependencies:
```shell
pip install -r requirements.txt
```

## Usage
Check the ipynb filees for example usage.

## Pruning Methods
- List the pruning methods supported by Smart-Pruner, such as:
  - Global Pruning
  - Random Unstructured Pruning
  - L1-Norm Based Filter pruning
  - Ln Structured pruning
  -  Pruning with 2:4 Sparsity (check research paper)
  -  Decay Pruning
  -  Thinet (check research paper)

## Architectures
- Specify the model-dataset architectures compatible with Smart-Pruner, for example:
  - ResNet-50 with CIFAR-10
  - AlexNet with CIFAR10
  - LeNet on MNIST
  - VggNet on CIFAR10

## Incorporating New Pruning Methods
- Smart-Pruner allows easy integration of new pruning methods. Follow these steps:
  1. Implement your new pruning method in the `/pruning` folder.
  2. Add the new pruning file name in init file of it.
  3. Import it in ipynb file.

## Contributing
We welcome contributions to Smart-Pruner! If you have suggestions, bug reports, or want to add new features, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
