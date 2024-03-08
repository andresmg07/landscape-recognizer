# landscape-recognizer
 
Supervised ML classification algorithm implemented through pattern recognition techniques and statistical methods.


![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project description

This project is focused on the design, implementation, and testing of a supervised machine learning algorithm aimed at **detecting urban-natural landscapes**. Leveraging pattern recognition techniques and statistical methods, the algorithm goal is to accurately logit classify images based on their color profile features. By harnessing the power of machine learning, this approach offers the potential to automate the identification process, enabling efficient analysis of large-scale datasets.

As result, the trained model is expected to be capable of determine with a high level of effectiveness if a given picture is an urban-natural landscape or not.  

### Constraints

1. The model must be trained with a limited dataset (12 samples).
2. The recognizer must rely on a model comprising only 3 features.
3. The features constituting the model must be exclusively extracted from the color profile of the images. The use of edge-based features is prohibited. 

## Algorithm design and implementation

In order to understand the problem to be solved consider the following definition: A picture catalogued as urban-natural landscapes must fulfill the following criteria:

1. The image depicts clear daytime skies or moderate cloud cover.
2. The horizon line is visible in the image.
3. The image portrays solid ground.
4. Bodies of water may or may not be visible in the image.

With the characterization provided above the feature vector is defined as follows:

$$ featureVector = (skyIdxArea, greenIdx, blueIdx $$


## Resulting model

## Testing

## Conclusion


## Author

**Andrés Montero Gamboa**<br>
Computing engineering undergraduate<br>
Instituto Tecnológico de Costa Rica<br>
[LinkedIn](https://www.linkedin.com/in/andres-montero-gamboa) | [GitHub](https://github.com/andresmg07)

## License

MIT License

Copyright (c) 2024 Andrés Montero Gamboa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.