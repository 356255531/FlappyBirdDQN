import tensorflow as tf
import numpy as np


class DQN_Flappy_Bird(object):
    """docstring for DQN_Flappy_Bird"""

    def __init__(self, nn_learning_rate):
        super(DQN_Flappy_Bird, self).__init__()
        self.input, self.output = self.create_network()
        self.label = tf.placeholder(tf.float32, shape=)

        self.saver = self.tf.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.input, labels=self.label
            )
        )

        self.optimizer = tf.train.AdamOptimizer(nn_learning_rate).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def __del__(self):
        self.sess.close()

    def create_network(self):
        def gen_weights_var(shape):
            inital = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(inital)

        def gen_bias_var(shape):
            inital = tf.constant(0.01, shape=shape)
            return tf.Variable(inital)

        def connect_conv2d(input, weights, stride):
            return tf.nn.conv2d(
                input,
                input,
                [1, stride, stride, 1],
                padding='SAME',
                use_cudnn_on_gpu=True
            )

        def connect_activ_relu(input, bias):
            return tf.nn.relu(input + bias)

        def connect_max_pool_2x2(intput):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 1st conv layer filter parameter
        weights_conv_1 = gen_weights_var([8, 8, 4, 32])
        bias_activ_conv_1 = gen_bias_var([32])

        # 2nd conv layer filter parameter
        weights_conv_2 = gen_weights_var([4, 4, 32, 64])
        bias_activ_conv_2 = gen_bias_var([64])

        # 3rd conv layer filter parameter
        weights_conv_3 = gen_weights_var([3, 3, 64, 64])
        bias_activ_conv_3 = gen_bias_var([64])

        # 4th fully connect net parameter
        weights_fc_layer_4 = gen_weights_var([1600, 512])
        bias_fc_layer_4 = gen_bias_var([512])

        # 5th fully connect net parameter
        weights_fc_layer_5 = gen_weights_var([512, 2])
        bias_fc_layer_5 = gen_bias_var([2])

        # input layer
        input_layer = tf.placeholder("float", [None, 80, 80, 4])

        # Convo layer 1
        output_conv_lay_1 = connect_conv2d(input_layer, weights_conv_1, 4)
        output_active_con_lay_1 = connect_activ_relu(output_conv_lay_1, bias_activ_conv_1)
        output_max_pool_layer_1 = connect_max_pool_2x2(output_active_con_lay_1)

        # Convo layer 2
        output_conv_lay_2 = connect_conv2d(output_max_pool_layer_1, weights_conv_2, 2)
        output_active_con_lay_2 = connect_activ_relu(output_conv_lay_2, bias_activ_conv_2)

        # Convo layer 3
        output_conv_lay_3 = connect_conv2d(output_active_con_lay_2, weights_conv_3, 1)
        output_active_con_lay_3 = connect_activ_relu(output_conv_lay_3, bias_activ_conv_3)
        output_reshape_layer_3 = tf.reshape(output_active_con_lay_3, [-1, 1600])  # Convo layer 3 reshape to fully connected net

        # Fully connect layer 4
        output_fc_layer_4 = tf.matmul(output_reshape_layer_3, weights_fc_layer_4)
        output_active_fc_layer_4 = connect_activ_relu(output_fc_layer_4, bias_fc_layer_4)

        # Output layer
        output = tf.matmul(output_active_fc_layer_4, weights_fc_layer_5) + bias_fc_layer_5

        return input_layer, output

    def save_weights(self, num_episode, saved_directory="./saved_weights"):
        self.saver.save(self.sess, saved_directory, global_step=num_episode)

    def load_weights(self, saved_directory="./saved_weights"):
        checkpoint = tf.train.get_checkpoint_state(saved_directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Weights successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find pre-trained network weights")

    def batch_parser(self):
        pass

    def batch_parser(self, batch):
        pass

    def train_network(self, batch, discount_factor, epsilon):
        # parse the batch input
        batch_input_frames, batch_input_label = self.batch_parser(batch)

        # train network
        self.sess.run(optimizer, feed_dict={
            self.input: batch_input_frames
        })

    def greedy_action_selection(self, frames):
        return np.argmax(self.sess(self.output, feed_dict={self.input=frames}))

    def epsilon_greedy_action_selection(self, frames, epsilon):
        if rd.sample() < epsilon:
            return rd.sample([0, 1])

        return self.greedy_action_selection(frames)
