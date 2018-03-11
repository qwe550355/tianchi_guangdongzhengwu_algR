# -*- coding:utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import copy as cp
import TensorflowUtils as utils
import argparse
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "50", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs_2017/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
URL = None

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224


def read_next_batch(batch_size, batch_num, data_base, shape):
    mat = np.zeros(shape, dtype=np.float32)
    for i in range(batch_size):
        mat[i] = (data_base[batch_num * batch_size + i])
    return mat


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            if name == 'conv1_1':
                kernels = utils.weight_variable([3, 3, 4, 64], name=name)
            else:
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            if name == 'conv5_1':
                current = utils.conv2d_atrous(current, kernels, bias)
            else:
                current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            if name == 'pool4' or name == 'pool5':
                current = utils.max_pool_notchange(current)
            else:
                current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image[:, :, :, :3], mean_pixel)
    pic_last = (tf.expand_dims(image[:, :, :, 3], -1) - 65.114) / 62.652
    processed_image = tf.concat([processed_image, pic_last], axis=3)
    # processed_image = tf.concat(3, [processed_image, tf.expand_dims(image[:, :, :, 3], -1)])
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_notchange(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 512], name="W6")
        b6 = utils.bias_variable([512], name="b6")
        conv6 = utils.conv2d_atrous(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 512, 512], name="W7")
        b7 = utils.bias_variable([512], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 512, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        # deconv_shape1 = image_net["pool4"].get_shape()
        # W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        # fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        #
        # deconv_shape2 = image_net["pool3"].get_shape()
        # W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        # b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        # fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(conv8, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    BATCH_SIZE = 46
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")  # drop_out
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 4], name="input_image")  # inputs
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name = "annotation")  # labels

    pred_annotation, logits = inference(image, keep_probability)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.." + str(ckpt.model_checkpoint_path) + "...")
    if FLAGS.mode == "train":
        imput_fcn = np.load('input_2017_for_predict_out_forgy_224.npy')
        label_fcn = np.load('label_2017_for_train_final_pic.npy')


        N_BATCH = len(label_fcn) // BATCH_SIZE
        for itr in range(31, MAX_ITERATION):
            for j in range(N_BATCH):
                train_images = read_next_batch(BATCH_SIZE, j, data_base=imput_fcn, shape=[BATCH_SIZE, 224, 224, 4])
                train_annotations = read_next_batch(BATCH_SIZE, j, data_base=label_fcn, shape=[BATCH_SIZE, 224, 224, 1])
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
                sess.run(train_op, feed_dict=feed_dict)
            train_images1 = read_next_batch(BATCH_SIZE, 0, data_base=imput_fcn, shape=[BATCH_SIZE, 224, 224, 4])
            train_annotations1 = read_next_batch(BATCH_SIZE, 0, data_base=label_fcn,
                                                shape=[BATCH_SIZE, 224, 224, 1])
            feed_dict1 = {image: train_images1, annotation: train_annotations1, keep_probability: 1.0}
            train_loss1, summary_str1 = sess.run([loss, summary_op], feed_dict=feed_dict1)
            print("Step: %d, Train_loss:%g" % (itr, train_loss1))
            if itr % 10 == 0:
                # saver.save(sess, URL.checkpointDir + "model.ckpt", itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            # if itr % 500 == 0:
            #     valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            #     valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
            #                                            keep_probability: 1.0})
            #     print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

    elif FLAGS.mode == "visualize":
        # inputs_data_2015 = (np.load('D:\jinda\\train\inputs_2015_split_224_224.npy'))
        inputs_data_2017 = (np.load('input_2017_for_predict_out_forgy_224.npy'))
        print(inputs_data_2017.shape)
        prediction_num = len(inputs_data_2017)
        # prediction_all_2015 = np.zeros((prediction_num, 224, 224, 1))
        prediction_all_2017 = np.zeros((prediction_num, 224, 224, 1))
        BATCH_SIZE_PREDICTION = 6

        # valid_annotations = np.squeeze(valid_annotations, axis=3)
        # for i in range(prediction_num // BATCH_SIZE_PREDICTION):
        #     batch_xs = read_next_batch(batch_size=BATCH_SIZE_PREDICTION, batch_num=i, data_base=inputs_data_2015,
        #                                shape=(BATCH_SIZE_PREDICTION, 224, 224, 3))
        #     prediction = sess.run(pred_annotation, feed_dict={image: batch_xs, keep_probability: 1.0})
        #     for j in range(BATCH_SIZE_PREDICTION):
        #         prediction_all_2015[j + i*BATCH_SIZE_PREDICTION] = cp.copy(prediction[j])
        for i in range(prediction_num // BATCH_SIZE_PREDICTION):
            batch_xs = read_next_batch(batch_size=BATCH_SIZE_PREDICTION, batch_num=i, data_base=inputs_data_2017,
                                       shape=(BATCH_SIZE_PREDICTION, 224, 224, 4))
            prediction = sess.run(pred_annotation, feed_dict={image: batch_xs, keep_probability: 1.0})
            for j in range(BATCH_SIZE_PREDICTION):
                prediction_all_2017[j + i*BATCH_SIZE_PREDICTION] = cp.copy(prediction[j])

        # np.save("D:\jinda\\train\\2015result_data.npy", prediction_all_2015)
        np.save("predict/result2017_logs_11111657.npy", prediction_all_2017)

9
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    URL, _ = parser.parse_known_args()
    tf.app.run()
