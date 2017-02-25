import tensorflow as tf

def w_var(shape):
    init = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init)

def b_var(shape):
    init = tf.constant(0.1, shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(images, keep_prob):
    with tf.name_scope('Convolutional Layer 1'):
        w_c1 = w_var([3,3,1,32])
        b_c1 = b_var([32])
        h_c1 = max_pool(tf.nn.relu(conv2d(images, w_c1) + b_c1))

    with tf.name_scope('Convolutional Layer 2'):
        w_c2 = w_var([5,5,32,32])
        b_c2 = b_var([64])
        h_c2 = max_pool(tf.nn.relu(conv2d(h_c1, w_c2) + b_c2))

    with tf.name_scope('Convolutional Layer 3'):
        w_c3 = w_var([5,5,32,64])
        b_c3 = b_var([128])
        h_c3 = max_pool(tf.nn.relu(conv2d(h_c2, w_c3) + b_c3))

    with tf.name_scope('Fully Connected Layer 1'):
        h_flat = tf.reshape(h_c3, [-1, 32*24*64])
        w_f1 = w_var([32*24*64, 4096])
        b_f1 = w_var([4096])
        h_f1 = tf.nn.relu(tf.matmul(h_flat, w_f1) + b_f1)

    with tf.name_scope('Dropout'):
        h_drop = tf.nn.dropout(h_f1, keep_prob)

    with tf.name_scope('Fully Connected Layer 2'):
        w_f2 = w_var([4096, 256])
        b_f2 = b_var([256])
        h_f2 = tf.nn.relu(tf.matmul(h_drop, w_f2) + b_f2)

    with tf.name_scope('Fully Connected Layer 3'):
        w_f3 = w_var([256,16])
        b_f3 = b_var([16])
        h_f3 = tf.nn.relu(tf.matmul(h_f2, w_f3) + b_f3)
    
    with tf.name_scope('Output'):
        w_f4 = w_var([16, 1])
        b_f4 = b_var([1])
        logit = tf.nn.relu(tf.matmul(h_f3, w_f4) + b_f4)

    return logit

def loss(label, logit):
    label = tf.to_float32(label)
    logit = tf.nn.sigmoid(logit)
    loss = tf.square(logit - label)
    return loss

def train(loss, lr):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(lr)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

