# Macular Hole Detection
We use a convolutional neural network for the detection and localization of macular holes.

##Architecture

### Image Preprocessing

CLAHE

Image resizing

Extract macula (Optional)

### Data Augmentation

Featurewise centering

Featurewise normalization

ZCA Whitening

Geometric shifts

### Convolutional Network

3x3 32 - ReLU - 2x2 Max Pool

3x3 32 - ReLU - 2x2 Max Pool

3x3 64 - ReLU - 2x2 Max Pool

3x3 128 - ReLU - 2x2 Max Pool

Dropout of 0.5

### Dense Layers

256 - ReLU - L2 reg 

Dropout of 0.5

32 - ReLU - L2 reg

Dropout of 0.5

1 - Sigmoid

### Loss Functions

Binary crossentropy

### Optimizer

Nestorov Adam -- RMSProp with Nesterov Momentum

### Information About Data

BATCH SIZE = 16

EPOCH      = 10

