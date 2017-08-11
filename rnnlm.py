import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in xrange(num_layers):
        cell = tf.contrib.rnn.BasicLSTMCell(H, forget_bias=0.0)
        cell = tf.contrib.rnn.DropoutWrapper(
                  cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        cells.append(cell)
        return tf.contrib.rnn.MultiRNNCell(cells)


# Decorator to avoid indentation troubles.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# This will save some indentation efforts, 
# and make things less error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. All the real graph
        construction is in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder_with_default(
                0.1, [], name="learning_rate")

            # For gradient clipping only.
            # Due to a bug in TensorFlow, this needs to be an ordinary
            # python constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph.

        This includes:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        The loss function is a *scalar* value that represents the
        _average_ loss across all samples in the batch
        """
        # Input ids, with dynamic shape depending on input.
        # With shape of [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

        # Initial hidden state. Initialize with cell.zero_state
        # once construct the RNN cell.
        self.initial_h_ = None

        # Final hidden state
        self.final_h_ = None

        # Output logits, which can be used by loss functions or for prediction.
        # Shape [batch_size, max_time]
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        # Get sequence length from input_w_.
        # A vector with elements ns[i] = len(input_w_[i])
        # Will be passed to dynamic_rnn.
        # Override this in feed_dict if want to have different-length sequences in the same batch
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

        # Construct embedding layer
        with tf.name_scope("Embedding_Layer"):
            self.W_in_ = tf.Variable(tf.random_uniform([self.V, self.H], -1.0, 1.0), name="embedding")
            self.x_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_)


        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("RNN_Layer"):
            self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
            self.initial_h_ = self.cell_.zero_state(self.batch_size_, tf.float32)
            self.rnn_outputs, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, self.x_, 
                                                           sequence_length=self.ns_,
                                                           initial_state=self.initial_h_)


        # Softmax output layer, over vocabulary. Compute logits_ here.
        with tf.name_scope("Output_Layer"):
            self.W_out_ = tf.Variable(tf.random_normal([self.H, self.V], -1.0, 1.0), name="W_out")
            self.b_out_ = tf.Variable(tf.zeros([self.V,], dtype=tf.float32), name="b_out")
            self.logits_ = tf.add(matmul3d(self.rnn_outputs, self.W_out_), self.b_out_, name="logits")


        # Loss computation (true loss, for prediction)
        with tf.name_scope("True_Loss"):
            self.per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_y_, 
                                                                               logits=self.logits_, 
                                                                               name="per_example_loss")
            self.loss_ = tf.reduce_mean(self.per_example_loss_, name="loss")


    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Loss function is a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. tf.reduce_mean).
        """
        # Training op
        self.train_step_ = None

        # Loss function
        self.train_loss_ = None

        # Define approximate loss function.
        # The number of samples is specified by self.softmax_ns.
        # Loss computation (sampled, for training)
        with tf.name_scope("Train_Loss"):
            self.per_example_train_loss_ = tf.nn.sampled_softmax_loss(weights=tf.transpose(self.W_out_), biases=self.b_out_,
                                             labels=tf.reshape(self.target_y_, [-1, 1]), 
                                             inputs=tf.reshape(self.rnn_outputs, [-1, self.H]),
                                             num_sampled=self.softmax_ns, num_classes=self.V,
                                             name="per_example_sampled_softmax_loss")
            self.train_loss_ = tf.reduce_mean(self.per_example_train_loss_, name="sampled_softmax_loss")


        # Define optimizer and training op
        with tf.name_scope("Training"):
            self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
            self.train_step_ = self.optimizer_.minimize(self.train_loss_)


    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        Define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.
        """

        self.pred_samples_ = tf.multinomial(tf.reshape(self.logits_, [-1, self.V]), 
                                            1, name="pred_samples")
        
        self.pred_samples_ = tf.reshape(self.pred_samples_, [self.batch_size_, self.max_time_, 1])


