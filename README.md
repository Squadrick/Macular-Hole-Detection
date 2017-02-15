# Macular-Hole-Detection
We use a convolutional neural network for the detection and localization of macular holes.

ARCHITECTURE

IMAGE PREPROCESSING
	1. CLAHE
	2. Image resizing
	3. Extract macula (Optional)

DATA AUGMENTATION
	1. Featurewise centering
	2. Featurewise normalization
	3. ZCA Whitening
	4. Geometric shifts

CONVNET
	1. 3x3 32 - ReLU - 2x2 Max Pool
	2. 3x3 32 - ReLU - 2x2 Max Pool
	3. 3x3 64 - ReLU - 2x2 Max Pool
	4. 3x3 128 - ReLU - 2x2 Max Pool
	4. Dropout of 0.5

DENSE LAYERS
	1. 256 - ReLU - L2 reg 
	2. Dropout of 0.5
	3. 32 - ReLU - L2 reg
	4. Dropout of 0.5
	5. 1 - Sigmoid

LOSS FUNCTION
	Binary crossentropy

OPTIMIZER
	Nestorov Adam -- RMSProp with Nesterov Momentum

BATCH SIZE = 16
EPOCH      = 10

