from model import *
from utils import *
import tensorflow as tf
from tqdm import tqdm, trange

"""
main function for training and parameter control
"""
batch_size = 50
# random generator dimension
z_dim = 100
# data directory
data_dir = "../../data/"
data_name = "img_align_celeba"

# get train and test data batch feeder
train_iter, train_init, test_iter, test_init = data_feeder2([data_dir + data_name], "jpg", batch_size, 0.1)

# hz2475
def build_graph(gp=False):
    """
    build computational graph, combining loss function, optimizer and gradient
    penality to train both generator and critic
    :param gp: weather to use gradient penalty or wight clipping (wGAN and improved wGAN)
    :return: optimizer for Generator and Critic
    """

    # sample noise to feed generator
    noise_dist = tf.contrib.distributions.Normal(0., 1.)
    z = noise_dist.sample((batch_size, z_dim))
    # generate "fake image" using generator, save in gen_data
    with tf.variable_scope('generator'):
        gen_data = generator(z)
    # placeholder for real image
    real_data = tf.placeholder(
        dtype=tf.float32,
        shape=((batch_size, 218, 178, 3)))
    # true_logic: critic's log probability for real image
    true_logit = critic(real_data)
    # true_logic: critic's log probability for generated image
    fake_logit = critic(gen_data, reuse=True)
    # c_loss moving_earth loss (discribed in wGAN) for discrimator
    c_loss = tf.reduce_mean(fake_logit - true_logit)
    # if using Gradient Penality for wGAN, calculate critic_loss's Gradient Penality
    if gp == True:
        # sample from uniform
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((batch_size, 1, 1, 1))
        # create interpolated data
        inter = real_data + alpha * (train - real_data)
        # calculate interpolated logit
        inter_logit = critic(inter, reuse=True)
        # calculate gradient for interpolated logit
        gradients = tf.gradients(inter_logit, [interpolated, ])[0]
        # calculate l2 distance
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        # calculate GP
        gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
        # summary for tensorboard
        gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
        grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
        c_loss += lam * gradient_penalty
    # use critic as generator loss
    g_loss = tf.reduce_mean(-fake_logit)
    # summary of loss on tensorboard
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    img_sum = tf.summary.image("gen", gen_data, max_outputs=10)
    # get variables for g and c for individually updating
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    # set optimizer
    opt_g = tf.train.RMSPropOptimizer(2e-4).minimize(g_loss, var_list=theta_g)
    opt_c = tf.train.RMSPropOptimizer(2e-4).minimize(c_loss, var_list=theta_c)
    # weight clipping for critic, clip weight to (-0.01,0.01)
    if gp == False:
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in theta_c]
        # merge the clip operations on critic variables
        with tf.control_dependencies([opt_c]):
            opt_c = tf.tuple(clipped_var_c)
    return opt_g, opt_c, real_data

# hz2475
def train(log_dir="logs", epoch=100, Citers=5):
    """
    train the network using built graph
    :param log_dir: <str> directory for tensorboard logging
    :param epoch: training epochs
    :param Citers: epochs for critic-only training
    """
    # build graph
    opt_g, opt_c, real_data = build_graph()
    # saver for checkpoints
    saver = tf.train.Saver()
    # merge summaries
    merged_all = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        t1 = trange(epoch, desc='1st loop')
        t2 = range(int(252559 / (batch_size) - 100))
        # epoch
        for e in t1:
            # initialize feeder
            sess.run(train_init)
            next_ele = train_iter.get_next()
            for step in t2:
                i = e * int(252559 / (batch_size * 12)) + step
                next_batch = sess.run(next_ele)
                step = step + 1
                # boost training for critic
                if i < 25 or i % 500 == 0:
                    citers = 50
                else:
                    citers = Citers
                # regular training for critic

                for j in range(citers):
                    next_batch = sess.run(next_ele)
                    step = step + 1
                    if i % 20 == 19 and j == 0:
                        # save summary every 20 steps
                        run_metadata = tf.RunMetadata()
                        _, merged = sess.run([opt_c, merged_all], feed_dict={real_data: next_batch})
                        summary_writer.add_summary(merged, i)
                        summary_writer.add_run_metadata(
                            run_metadata, 'critic_metadata {}'.format(i), i)
                    else:
                        sess.run(opt_c, feed_dict={real_data: next_batch})
                next_batch = sess.run(next_ele)
                step = step + 1
                if i % 20 == 19:
                    # save summary every 20 steps
                    _, merged = sess.run([opt_g, merged_all], feed_dict={real_data: next_batch})
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'generator_metadata {}'.format(i), i)
                else:
                    # update generator
                    sess.run(opt_g, feed_dict={real_data: next_batch})
                # save weight
                if i % 1000 == 999:
                    saver.save(sess, os.path.join(
                        ckpt_dir, "model.ckpt"), global_step=i)

# train the model
train(log_dir="logs", epoch=100, Citers=5)
