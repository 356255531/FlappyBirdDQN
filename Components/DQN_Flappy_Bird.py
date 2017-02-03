import tensorflow as tf


class DQN_Flappy_Bird(object):
    """docstring for DQN_Flappy_Bird"""

    def __init__(self,
                 batch_size,
                 patch_size,
                 num_channels,
                 depth,
                 image_size
                 ):
        super(DQN_Flappy_Bird, self).__init__()
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.depth = depth
        self.image_size = image_size

        self.create_network()
        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph)
        self.init_parameter()

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

    def save_weights(self, saver, saved_directory="saved_weights", num_episode):
        saver.save(self.sess, saved_directory + "dqn_weights", global_step=num_episode)

    def load_weights(self, saver, saved_directory="saved_weights"):
        checkpoint = tf.train.get_checkpoint_state(saved_directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Weights successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find pre-trained network weights")

    def predict(self):
        pass
