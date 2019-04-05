import tensorflow as tf
from utils import check_and_create_dir, print_train_steps, get_batch, extract_image_path, extract_n_normalize_image
from PIL import Image
import os
import numpy as np
import cv2
from scipy.misc import imsave

class AELikeModel:
    """
    AE-like Model with Pooling as a Size-changing Factor
    """
    def __init__(self, image_size, alpha, AE = True,verbose=False, trained_model=None):
        # self.Y is the NN

        tf.reset_default_graph()
        self.image_size = image_size
        self.alpha = alpha
        self.verbose = verbose
        self.X = tf.placeholder(tf.float32,shape =  [None, self.image_size, self.image_size, 1])
        self.Y_clear = tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size, 1])

        if AE:
            n_filters = [16, 32, 64]
            filter_sizes = [5, 5, 5]
        else:
            n_filters = [16,32,64,128,256]
            filter_sizes = [5, 5, 5, 5, 5]

        n_input = 1

        Ws = []
        shapes = []

        current_input = self.X
        
        # encoding convolutional layers
        for layer_i, n_output in enumerate(n_filters):
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                shapes.append(current_input.get_shape().as_list())
                W = tf.get_variable(
                    name='W',
                    shape=[
                        filter_sizes[layer_i],
                        filter_sizes[layer_i],
                        n_input,
                        n_output],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
                h = tf.nn.conv2d(current_input, W,
                    strides=[1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(h)
                if AE:
                    current_input = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
                else:
                    current_input = tf.nn.max_pool(conv, [1,1,1,1], [1,1,1,1], padding='SAME')
                Ws.append(W)
                n_input = n_output

        # decoding convultional layers (mirrored and reversed encoding layer)
        # skipped if Family CNN model
        if AE:
            Ws.reverse()
            shapes.reverse()
            n_filters.reverse()
            n_filters = n_filters[1:] + [1]

            for layer_i, shape in enumerate(shapes):
                with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                    W = Ws[layer_i]
                    h = tf.nn.conv2d_transpose(current_input, W,
                        tf.stack([tf.shape(self.X)[0], shape[1], shape[2], shape[3]]),
                        strides=[1, 2, 2, 1], padding='SAME')
                    current_input = tf.nn.relu(h)

                    
        # self.Y is combined function on all layers taking in self.X
        self.Y = current_input

        # MSE
        self.mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.Y_clear, self.Y), 1))

        # MS SSIM (minimizes large scale changes, see readMe)
        self.ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(self.Y_clear, self.Y, 1))
        # Mixed cost
        self.cost = self.alpha*self.ssim + (1 - self.alpha)*self.mse

        # Using Adam for optimizer
        self.learning_rate = tf.Variable(initial_value=1e-2, trainable=False, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.batch_size = tf.Variable(initial_value=64, trainable=False, dtype=tf.int32)
        self.trained_model = trained_model

    def init_session(self):
        """
        Init session
        returns saved model if it exists
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(sess=sess, coord=coord)

        if not self.trained_model is None:
            saver.restore(sess, self.trained_model)
        return (sess,saver)

    def train(self, x_path_dir, y_path_dir, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_model, output_log, b_size):
        """
        Train data
        """
        # Check output directory
        # suffix for clafification on type
        if output_model:
            output_model+="AE"
        else:
            output_model+="MultiCNN"

        check_and_create_dir(output_model)

        # Load data
        x_filenames = extract_image_path([x_path_dir])
        y_filenames = extract_image_path([y_path_dir])

        # Scalar
        tf.summary.scalar('Learning rate', self.learning_rate)
        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('MS SSIM', self.ssim)
        tf.summary.scalar('Loss', self.cost)
        tf.summary.image('BSE', self.Y)
        tf.summary.image('Ground truth', self.Y_clear)
        merged = tf.summary.merge_all()

        sess, saver = self.init_session()
        writer = tf.summary.FileWriter(output_log, sess.graph)

        l_rate = learning_rate
        try:
            for epoch_i in range(epochs):
                if ((epoch_i + 1) % epochs_to_reduce_lr) == 0:
                    l_rate = l_rate * (1 - reduce_lr)
                if self.verbose:
                    print("\n------------ Epoch : ",epoch_i+1)
                    print("Current learning rate {}".format(l_rate))

                # Training steps
                for i in range(train_steps):
                    if self.verbose:
                        print_train_steps(i+1, train_steps)
                    x_batch, y_batch = get_batch(b_size, self.image_size, x_filenames, y_filenames)

                    sess.run(self.optimizer, feed_dict={ self.X: x_batch, self.Y_clear: y_batch, self.learning_rate: l_rate, self.batch_size: b_size })

                    if i % 50 == 0:
                        summary = sess.run(merged, {self.X: x_batch, self.Y_clear: y_batch, self.learning_rate: l_rate, self.batch_size: b_size})
                        writer.add_summary(summary, i+ epoch_i*train_steps)
                if self.verbose:
                    print("\nSave model to {}".format(output_model))
                saver.save(sess, output_model, global_step=(epoch_i+1)*train_steps)
        except KeyboardInterrupt:
            saver.save(sess, output_model)

    def suppress(self, input_image, output_path):
        '''
            suppresses bones in an individual image using trained model
        '''

        img = extract_n_normalize_image(input_image)
        
        img = cv2.resize(img,(self.image_size,self.image_size))
        # print(img.shape)
        x_image = np.reshape(img, (1, self.image_size, self.image_size, 1))
        # x_image = np.reshape(np.array([img]), (1, self.image_size, self.image_size, 1))
        sess, _ = self.init_session()
        y_image = sess.run(self.Y, feed_dict={self.X: x_image})
        encoded_image = y_image.reshape((self.image_size, self.image_size)) 
        location = output_path+"/" +input_image
        imsave(output_path, encoded_image)


    def test(self, X_path, Y_path, save_output=False):
        '''
        Test output of model on new batch of inputs versus ground truth
        saves output of model as image and saves difference between pixelwise difference as image
        '''

        # produces list of file names from directory paths
        print(save_output, "print")
        X = extract_image_path([X_path])
        Y = extract_image_path([X_path])

        sess, _ = self.init_session()

        # scalar errors for each image
        MSE = np.zeros((len(X)))
        SSIM = np.zeros((len(X)))
        cost = np.zeros((len(X)))

        # combine images into one error image
        total_signed_error_image = np.zeros((self.image_size,self.image_size))
        total_abs_error_image = np.zeros((self.image_size,self.image_size))
        total_MSE_image = np.zeros((self.image_size,self.image_size))


        # extracts and predicts one image at a time
        for i in range(len(X)):

            input_image = extract_n_normalize_image(X[i])

            # dimensions are (batch size, height, width, RGB channels)
            x_image = np.reshape(np.array([input_image]), (1, self.image_size, self.image_size, 1))
            y_image = np.reshape(np.array([input_image]), (1, self.image_size, self.image_size, 1))
            output_image = sess.run(self.Y, feed_dict = {self.X: x_image})

            y_truth = extract_n_normalize_image(Y[i])
            # print(output_image.shape, "y_truth")
            # erases trivial dimensions from output
            y_pred = np.reshape(np.array([output_image]), (self.image_size, self.image_size))

            error_image = y_truth-y_pred
            abs_error_image = np.absolute(error_image)
            squared_error_image = (np.square(error_image))
            signed_error_image = error_image

            total_abs_error_image += abs_error_image
            total_signed_error_image += signed_error_image
            total_MSE_image += squared_error_image
            
            # SSIM[i] = tf.image.ssim_multiscale(Image.fromarray(y_image[0,:,:,0]), Image.fromarray(y_image[0,:,:,0]), 1)
            # SSIM[i] = tf.image.ssim_multiscale(y_image[0,:,:,0], y_image[0,:,:,0], 1)
            MSE[i] = np.average(squared_error_image)
            print(MSE[i])
            cost[i] = self.alpha*SSIM[i] + (1 - self.alpha)*MSE[i]
            

            if save_output:
                imsave(save_output+str(i)+"_true.png", y_pred )
                # imsave(save_output+str(i)+"_mse", mse)

        if save_output:
            imsave(save_output+"comb_MSE.png", total_MSE_image)
            imsave(save_output+"comb_SignedError.png", total_signed_error_image/leg)

        # return {"cost": np.average(cost),"MSE": np.average(MSE),"SSIM": np.average(SSIM),"MSE_IMAGE": total_MSE_image/len(X),"ABS_ERROR_IMAGE": abs_error_image/len(X), "SIGNED_ERROR_IMAGE": total_signed_error_image/len(X) }
        return {"cost": np.average(cost),"MSE": np.average(MSE),"MSE_IMAGE": total_MSE_image/len(X),"ABS_ERROR_IMAGE": abs_error_image/len(X), "SIGNED_ERROR_IMAGE": total_signed_error_image/len(X) }








