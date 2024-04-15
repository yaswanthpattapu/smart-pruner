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


## Usage
1. Navigate to the Smart-Pruner directory:
   ```shell
   cd Smart-Pruner
   ```
2. Run the Smart-Pruner module with your desired configurations:
   ```shell
   python smart_pruner.py --model <model_name> --dataset <dataset_name> --pruning_method <method_name>
   ```

## Pruning Methods
- List the pruning methods supported by Smart-Pruner, such as:
  - Iterative Pruning
  - Magnitude Pruning
  - Weight Rewinding
  - ...

## Architectures
- Specify the model-dataset architectures compatible with Smart-Pruner, for example:
  - ResNet-50 with CIFAR-10
  - MobileNetV2 with ImageNet
  - ...

## Incorporating New Pruning Methods
- Smart-Pruner allows easy integration of new pruning methods. Follow these steps:
  1. Implement your new pruning method in the `pruning_methods.py` file.
  2. Add the method to the list of supported methods in `smart_pruner.py`.
  3. Run Smart-Pruner with the new method using the `--pruning_method` argument.

## Contributing
We welcome contributions to Smart-Pruner! If you have suggestions, bug reports, or want to add new features, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
