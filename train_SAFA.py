from models import SAFA
from dataloader import DataLoader
import tensorflow as tf
import numpy as np
import os
import time


# -------------------------------------------------------- #
def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def validate_top(grd_descriptor, sat_descriptor, dataloader):
    accuracy = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))

    top1_percent = int(dist_array.shape[1] * 0.01) + 1
    top1 = 1
    top5 = 5
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, dataloader.test_label[i][0]]
        prediction = np.sum(dist_array[i, :] < gt_dist)

        dist_temp = np.ones(dist_array[i, :].shape[0])
        dist_temp[dataloader.test_label[i][1:]] = 0
        prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

        if prediction < top1_percent:
            accuracy += 1.0
        if prediction < top1:
            accuracy_top1 += 1.0
        if prediction < top5:
            accuracy_top5 += 1.0
        if prediction_hit < top1:
            accuracy_hit += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    accuracy_top1 /= data_amount
    accuracy_top5 /= data_amount
    accuracy_hit /= data_amount
    return accuracy, accuracy_top1, accuracy_top5, accuracy_hit


def compute_loss(sat_global, grd_global, batch_hard_count=0, batch_size = 14, loss_weight = 10.0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=1, mode='', model_dir='', load_model_path='', load_mining_path='', break_iter=None,
          dim=4096, number_of_epoch = 30, learning_rate_val = 1e-5, batch_size = 14):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''
    # import data
    dataloader = DataLoader(mode=mode, dim=dim, same_area=True if 'same' in mode else False)
    if break_iter is None:
        break_iter = int(dataloader.train_data_size / batch_size)

    print('batch_size:',batch_size, 'break_iter:',break_iter)

    sat_x = tf.placeholder(tf.float32, [None, dataloader.sat_size[0], dataloader.sat_size[1], 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, dataloader.grd_size[0], dataloader.grd_size[1], 3], name='grd_x')

    global_step = tf.Variable(0, trainable=False)

    # build model
    sat_global, grd_global, sat_return, grd_return = SAFA(sat_x, grd_x)

    # choose loss
    loss = compute_loss(sat_global, grd_global, batch_hard_count=0,batch_size=batch_size)

    # set training
    rate = tf.train.exponential_decay(learning_rate_val, global_step, 150000, 0.1)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    var_list_restore = [v for v in tf.trainable_variables() if 'Gap' not in v.name]

    if start_epoch == 1:
        saver = tf.train.Saver(var_list_restore, max_to_keep=None)
    else:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if start_epoch != 1:
            print('load model...')
            saver.restore(sess, os.path.join(load_model_path , 'model.ckpt'))
            print("   Model loaded from: %s" % load_model_path)
            print('load model...FINISHED')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                t1 = time.time()
                # train
                batch_sat, batch_grd, batch_list, delta_list = dataloader.get_next_batch(batch_size)
                if batch_sat is None or iter == break_iter:
                    break
                global_step_val = tf.train.global_step(sess, global_step)
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd}

                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    t2 = time.time()
                    print('global %d, epoch %d, iter %d: loss : %.4f, time: %.4f' % (global_step_val, epoch, iter, loss_val, t2- t1))
                else:
                    _ = sess.run([train_step], feed_dict=feed_dict)

                iter += 1
            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            dataloader.reset_scan()
            sat_global_descriptor = np.zeros([dataloader.test_sat_data_size, dim])
            grd_global_descriptor = np.zeros([dataloader.test_data_size, dim])

            # compute sat descriptors
            val_i = 0
            while True:
                if val_i % 100 == 0:
                    print('sat      progress %d' % val_i)

                batch_sat = dataloader.next_sat_scan(64)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat}

                sat_global_val = sess.run(sat_global, feed_dict=feed_dict)
                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                val_i += sat_global_val.shape[0]

            # compute grd descriptors
            val_i = 0
            while True:
                if val_i % 100 == 0:
                    print('grd      progress %d' % val_i)
                batch_grd = dataloader.next_grd_scan(32)
                if batch_grd is None:
                    break
                feed_dict = {grd_x: batch_grd}
                grd_global_val = sess.run(grd_global, feed_dict=feed_dict)
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += grd_global_val.shape[0]

            print('   compute accuracy')
            val_accuracy, val_accuracy_top1, val_accuracy_top5, hit_rate = validate_top(grd_global_descriptor,
                                                                              sat_global_descriptor, dataloader)
            print('Evaluation epoch %d: accuracy = %.1f%% , top1: %.1f%%, top5: %.1f%%, hit_rate: %.1f%%' % (
            epoch, val_accuracy * 100.0, val_accuracy_top1 * 100.0, val_accuracy_top5 * 100.0, hit_rate * 100.0))

            model_save_dir = os.path.join(model_dir, str(epoch))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            np.save(os.path.join(model_dir, 'sat_global_descriptor.npy'), sat_global_descriptor)
            np.save(os.path.join(model_dir, 'grd_global_descriptor.npy'), grd_global_descriptor)
            save_path = saver.save(sess, os.path.join(model_save_dir, 'model.ckpt'))
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            print("Model saved in file: {}".format(save_path))
            # ---------------------------------------------------------


if __name__ == '__main__':
    gpu_visible = "0"
    mode = 'train_SAFA_CVM-loss-same'
    start_epoch = 1
    mining_start = -1
    number_of_epoch = 30
    learning_rate_val = 1e-5
    batch_size = 14
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
    model_load_dir = ''
    model_save_dir = './data/'
    load_mining_path = ''

    train(start_epoch=start_epoch, mode=mode, model_dir=model_save_dir, load_model_path=model_load_dir,
          number_of_epoch=number_of_epoch, learning_rate_val=learning_rate_val, batch_size=batch_size)

