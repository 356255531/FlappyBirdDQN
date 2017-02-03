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

        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph)
        self.init_parameter()

    def gen_weights_var(self, shape):
        inital = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(inital)

    def gen_bias_var(self, shape):
        inital = tf.constant(0.01, shape=shape)
        return tf.Variable(inital)

    def create_network(self):
        # 1st conv layer filter parameter
        weights_conv_1 = self.gen_weights_var([8, 8, 4, 32])
        # 1st conv layer Activation bias
        bias_activ_conv_1 = self.gen_bias_var([32])

        # 2nd conv layer filter parameter
        weights_conv_2 = self.gen_weights_var([4, 4, 32, 64])
        # 2nd conv layer activation bias
        bias_activ_conv_2 = self.gen_bias_var([64])

        # 3rd conv layer filter parameter
        weights_conv_3 = self.gen_weights_var([3, 3, 64, 64])
        # 3rd conv layer activation bias
        bias_activ_conv_3 = self.gen_bias_var([64])

    def save_weights(self, saver, saved_directory="saved_weights", num_episode):
        saver.save(self.sess, saved_directory + "dqn_weights", global_step=num_episode)

    def load_weights(self, saver, saved_directory="saved_weights"):
        checkpoint = tf.train.get_checkpoint_state(saved_directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Weights successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find pre-trained network weights")

    def forward_propagate(self):
        pass

    def function():
        pass

    def init_parameter(self):
        # Conv Layer 1
        self.layer_1_conv_filter = tf.Variable(tf.truncated_normal(
            [self.patch_size[0],
             self.patch_size[0],
             4,  # Number of Channel
             self.num_conv_kern_conv_layer[0]
             ],
            stddev=0.1
        ))
        self.layer_1_conv_biases = tf.Variable(tf.zeros(
            [4 * self.num_conv_kern_conv_layer[0]]
        ))

        # Conv Layer 2
        num_channels = 4 * self.num_conv_kern_conv_layer[0]
        self.layer_2_conv_filter = tf.Variable(tf.truncated_normal(
            [self.patch_size[1],
             self.patch_size[1],
             num_channels,
             self.num_conv_kern_conv_layer[1]
             ],
            stddev=0.1
        ))
        self.layer_2_conv_biases = tf.Variable(tf.constant(1.0, shape=[
            num_channels * self.num_conv_kern_conv_layer[1]
        ]))

        # Conv Layer 3
        num_channels *= self.num_conv_kern_conv_layer[1]
        self.layer_3_conv_filter = tf.Variable(tf.truncated_normal(
            [self.patch_size[2],
             self.patch_size[2],
             num_channels,
             self.num_conv_kern_conv_layer[2]
             ],
            stddev=0.1
        ))
        self.layer_3_conv_biases = tf.Variable(tf.constant(1.0, shape=[
            num_channels * self.num_conv_kern_conv_layer[2]
        ]))

        # full connect hiden layer
        num_channels *= self.num_conv_kern_conv_layer[2]
        self.layer_4_hidden_weights = tf.Variable(tf.truncated_normal(
            [(self.image_size // 4) * (self.image_size // 4) * num_channels, self.num_hidden], stddev=0.1))
        self.layer_4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))

        # matrix multilpication
        self.layer_4_weights = tf.Variable(tf.truncated_normal(
            [self.num_hidden, self.num_actions], stddev=0.1))
        self.layer_4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_actions]))

    def mode(self, frame_4):
        # calculate the 1st conv layer output
        layer_1_conv_output = tf.nn.conv2d(
            frame_4,
            self.layer_1_conv_filter,
            [1, self.conv_stride[0], self.conv_stride[0], 1],
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        max_pooling_1_output = tf.nn.max_pool(
            layer_1_conv_output,
            [1, self.max_pool_kernel_size[0], self.max_pool_kernel_size[0], 1],
            [1, self.max_pool_stride[0], self.max_pool_stride[0], 1],
            padding='SAME'
        )
        layer_1_output = tf.nn.relu(max_pooling_1_output + self.layer_1_conv_biases)

        # calculate the 2nd conv layer output
        layer_2_conv_output = tf.nn.conv2d(
            layer_1_output,
            self.layer_2_conv_filter,
            [1, self.conv_stride[0], self.conv_stride[0], 1],
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        max_pooling_2_output = tf.nn.max_pool(
            layer_2_conv_output,
            [1, self.max_pool_kernel_size[1], self.max_pool_kernel_size[1], 1],
            [1, self.max_pool_stride[1], self.max_pool_stride[1], 1],
            padding='SAME'
        )

        # calculate the 3rd conv layer output
        layer_3_conv_output = tf.nn.conv2d(
            layer_2_output,
            self.layer_2_conv_filter,
            [1, self.conv_stride[0], self.conv_stride[0], 1],
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        max_pooling_2_output = tf.nn.max_pool(
            layer_2_conv_output,
            [1, self.max_pool_kernel_size[1], self.max_pool_kernel_size[1], 1],
            [1, self.max_pool_stride[1], self.max_pool_stride[1], 1],
            padding='SAME'
        )

        layer_2_output = tf.nn.relu(max_pooling_2_output + layer_2_conv_biases)
        shape = layer_2_output.get_shape().as_list()
        reshape = tf.reshape(layer_2_output, [shape[0], shape[1] * shape[2] * shape[3]])
        layer_3_output = tf.nn.relu(tf.matmul(reshape, layer_3_weights) + layer_3_biases)
        return tf.matmul(layer_3_output, layer_4_weights) + layer_4_biases

    def predict(self, frame_4):
        return self.model(frame_4)

    def model(self, frame_4):
        layer_1_conv_output = tf.nn.conv2d(
            data,
            layer_1_conv_filter,
            [1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        max_pooling_1_output = tf.nn.max_pool(
            layer_1_conv_output,
            [1, max_pool_kernel_size, max_pool_kernel_size, 1],
            [1, max_pool_stride, max_pool_stride, 1],
            padding='SAME'
        )
        layer_1_output = tf.nn.relu(max_pooling_1_output + layer_1_conv_biases)
        layer_2_conv_output = tf.nn.conv2d(
            layer_1_output,
            layer_2_conv_filter,
            [1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True
        )
        max_pooling_2_output = tf.nn.max_pool(
            layer_2_conv_output,
            [1, max_pool_kernel_size, max_pool_kernel_size, 1],
            [1, max_pool_stride, max_pool_stride, 1],
            padding='SAME'
        )
        layer_2_output = tf.nn.relu(max_pooling_2_output + layer_2_conv_biases)
        shape = layer_2_output.get_shape().as_list()
        reshape = tf.reshape(layer_2_output, [shape[0], shape[1] * shape[2] * shape[3]])
        layer_3_output = tf.nn.relu(tf.matmul(reshape, layer_3_weights) + layer_3_biases)
        return tf.matmul(layer_3_output, layer_4_weights) + layer_4_biases
