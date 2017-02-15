# Macular-Hole-Detection
We use a convolutional neural network for the detection and localization of macular holes.

##ARCHITECTURE

### IMAGE PREPROCESSING
CLAHE
Image resizing
Extract macula (Optional)

### DATA AUGMENTATION
Featurewise centering
Featurewise normalization
ZCA Whitening
Geometric shifts

### CONVNET
3x3 32 - ReLU - 2x2 Max Pool
3x3 32 - ReLU - 2x2 Max Pool
3x3 64 - ReLU - 2x2 Max Pool
3x3 128 - ReLU - 2x2 Max Pool
Dropout of 0.5

### DENSE LAYERS
256 - ReLU - L2 reg 
Dropout of 0.5

32 - ReLU - L2 reg
Dropout of 0.5

1 - Sigmoid

### LOSS FUNCTION
Binary crossentropy

### OPTIMIZER
Nestorov Adam -- RMSProp with Nesterov Momentum

BATCH SIZE = 16
EPOCH      = 10

