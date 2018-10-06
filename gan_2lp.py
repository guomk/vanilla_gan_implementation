import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def plot(samples):
    """Data ploting function"""
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Discrimination net
X = tf.placeholder(tf.float32, shape=[None, 784], name='x')

D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_D2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]
# Note that we are using the same network for both the discriminator and generator


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, G_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)  # The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function.

    return D_prob, D_logit


def generator(z):
    # Takes a 100-dim vector and returns 786-dim vector, which is a 28x28 MNIST image
    # z comes from a prior space, we are learning a mapping form the prior space to Pdata (distribution of data)
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


# Define the adversarial process
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# In the following statements we add the negative sign and slightly modified
# the lost function for generator because tensorflow can only perform minimize operation
# in our case we want it maximized
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# As the paper suggested, it is a better practice to
# maximize tf.reduce_mean(tf.log(D_fake))
# instead of minimizing tf.reduce_mean(1 - tf.log(D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))


# Then we train the network one by one with adversarial training

# Only update D(X)'s param, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s param, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0


def sample_Z(m, n):
    """Uniform prior for G(Z)"""
    return np.random.uniform(-1., 1., size=[m, n])





for it in range(100000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={
            Z: sample_Z(16, Z_dim)  # Sample 16 datas form prior to observe output
        })
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
