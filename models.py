from backbone.VGG import Vgg16
import tensorflow as tf
import loupe as lp

# the clean siamese network used in https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_Revisiting_Street-to-Aerial_View_Image_Geo-Localization_and_Orientation_Estimation_WACV_2021_paper.pdf
def clean_siamese(x_sat, x_grd, keep_prob, cam=False):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)

    with tf.variable_scope('vgg_sat'):
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(x_sat)

    sat_global, grd_global, _, _, fc_sat, fc_grd = siamese_gap_fc(sat_local, grd_local, 'Gap_local', keep_prob)

    # for visualization
    if cam:
        return sat_global, grd_global, sat_local, grd_local, fc_sat, fc_grd

    return sat_global, grd_global, sat_local, grd_local


# CVMNet https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.pdf
def siamese_vlad(x_sat, x_grd, keep_prob):
    with tf.device('/gpu:0'):
        with tf.variable_scope('vgg_grd'):
            vgg_grd = Vgg16()
            grd_local = vgg_grd.build(x_grd)

        with tf.variable_scope('netvlad_grd'):
            netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(grd_local)[1] * tf.shape(grd_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=True)
            grd_vlad = netvlad_grd.forward(grd_local)
        with tf.variable_scope('vgg_sat'):
            vgg_sat = Vgg16()
            sat_local = vgg_sat.build(x_sat)

        with tf.variable_scope('netvlad_sat'):
            netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_local)[1] * tf.shape(sat_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=True)
            sat_vlad = netvlad_sat.forward(sat_local)

    with tf.device('/gpu:1'):
        sat_global, grd_global = siamese_fc(sat_vlad, grd_vlad, 'dim_reduction', keep_prob)
    return sat_global, grd_global, sat_local, grd_local


# =================================================================================================================
# SAFA: https://papers.nips.cc/paper/2019/file/ba2f0015122a5955f8b3a50240fb91b2-Paper.pdf
def SAFA(x_sat, x_grd, dimension=8, trainable=True, original=False):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)
        grd_local_out = grd_local
        grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension * channel])

    with tf.variable_scope('vgg_sat'):
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(x_sat)
        sat_local_out = sat_local
        sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension * channel])
    if original:
        return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global,
                                                                         dim=1), sat_global, grd_global, sat_local_out, grd_local_out
    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1), sat_local, grd_local


# SAFA with semi-positive reference
def SAFA_semi(x_sat, x_sat_semi, x_grd, dimension=8, trainable=True):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)
        grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension * channel])

    with tf.variable_scope('vgg_sat') as scope:
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(tf.concat([x_sat, x_sat_semi], axis=0))
        sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension * channel])

    sat_global_split_1, sat_global_split_2 = tf.split(sat_global, 2, axis=0)

    return tf.nn.l2_normalize(sat_global_split_1, dim=1), tf.nn.l2_normalize(sat_global_split_2, dim=1),\
           tf.nn.l2_normalize(grd_global, dim=1), sat_local, grd_local


# SAFA with offset prediction
def SAFA_delta(x_sat, x_sat_semi, x_grd, dimension=8, trainable=True, out_dim=2):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)
        grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension * channel])

    with tf.variable_scope('vgg_sat') as scope:
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(tf.concat([x_sat, x_sat_semi], axis=0))
        sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension * channel])

    sat_global_split_1, sat_global_split_2 = tf.split(sat_global, 2, axis=0)

    both_feature = tf.concat([sat_global_split_1, grd_global], axis=-1)

    both_feature_fc = fc_layer(both_feature, 4096 * 2, 512, 0.005, 0.1, 'fc_delta_1')
    delta = fc_layer(both_feature_fc, 512, out_dim, 0.005, 0.1, 'fc_delta_2', activation_fn=None)

    return tf.nn.l2_normalize(sat_global_split_1, dim=1), tf.nn.l2_normalize(sat_global_split_2, dim=1),\
           tf.nn.l2_normalize(grd_global, dim=1), sat_local, grd_local, delta


# for validation, no semi-positive input
def SAFA_delta_validate(x_sat, x_grd, dimension=8, trainable=True, out_dim=2):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)
        grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension * channel])

    with tf.variable_scope('vgg_sat') as scope:
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(x_sat)
        sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension * channel])

    both_feature = tf.concat([sat_global, grd_global], axis=-1)

    both_feature_fc = fc_layer(both_feature, 4096 * 2, 512, 0.005, 0.1, 'fc_delta_1')
    delta = fc_layer(both_feature_fc, 512, out_dim, 0.005, 0.1, 'fc_delta_2', activation_fn=None)

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1), sat_local, grd_local, delta


# =================================================================================================================
# simple siamese VGG with offset (delta) prediction, used when there is no orientation alignment and safa does not work
def clean_siamese_delta(x_sat, x_sat_semi, x_grd, out_dim=2):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)

    with tf.variable_scope('vgg_sat'):
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(tf.concat([x_sat, x_sat_semi], axis=0))

    with tf.variable_scope('Gap_local') as scope:
        x_sat = tf.reduce_mean(sat_local, axis=[1, 2])
        x_grd = tf.reduce_mean(grd_local, axis=[1, 2])

        fc_sat = fc_layer(x_sat, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        # sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

        scope.reuse_variables()

        fc_grd = fc_layer(x_grd, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)

    fc_sat_split_1, fc_sat_split_2 = tf.split(fc_sat, 2, axis=0)

    both_feature = tf.concat([fc_sat_split_1, fc_grd], axis=-1)

    both_feature_fc = fc_layer(both_feature, 4096 * 2, 512, 0.005, 0.1, 'fc_delta_1')
    delta = fc_layer(both_feature_fc, 512, out_dim, 0.005, 0.1, 'fc_delta_2', activation_fn=None)

    return tf.nn.l2_normalize(fc_sat_split_1, dim=1), tf.nn.l2_normalize(fc_sat_split_2, dim=1), tf.nn.l2_normalize(
        fc_grd, dim=1), sat_local, grd_local, delta


def clean_siamese_delta_validate(x_sat, x_grd, out_dim=2):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        grd_local = vgg_grd.build(x_grd)

    with tf.variable_scope('vgg_sat'):
        vgg_sat = Vgg16()
        sat_local = vgg_sat.build(x_sat)

    with tf.variable_scope('Gap_local') as scope:
        x_sat = tf.reduce_mean(sat_local, axis=[1, 2])
        x_grd = tf.reduce_mean(grd_local, axis=[1, 2])

        fc_sat = fc_layer(x_sat, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        # sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

        scope.reuse_variables()

        fc_grd = fc_layer(x_grd, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)

    both_feature = tf.concat([fc_sat, fc_grd], axis=-1)

    both_feature_fc = fc_layer(both_feature, 4096 * 2, 512, 0.005, 0.1, 'fc_delta_1')
    delta = fc_layer(both_feature_fc, 512, out_dim, 0.005, 0.1, 'fc_delta_2', activation_fn=None)

    return tf.nn.l2_normalize(fc_sat, dim=1), tf.nn.l2_normalize(fc_grd, dim=1), sat_local, grd_local, delta


# ================================================================================================
# siamese head with shared fc layers
def siamese_gap_fc(sat_local, grd_local, scope_name, dropout=0.5):
    with tf.variable_scope(scope_name) as scope:
        x_sat = tf.reduce_mean(sat_local, axis=[1, 2])
        x_grd = tf.reduce_mean(grd_local, axis=[1, 2])

        x_sat_drop = tf.nn.dropout(x_sat, dropout, name='sat_dropout')
        fc_sat = fc_layer(x_sat_drop, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

        scope.reuse_variables()

        x_grd_drop = tf.nn.dropout(x_grd, dropout, name='grd_dropout')
        fc_grd = fc_layer(x_grd_drop, 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        grd_global = tf.nn.l2_normalize(fc_grd, dim=1)

    return sat_global, grd_global, x_sat, x_grd, fc_sat, fc_grd


def siamese_fc(sat_local, grd_local, scope_name, dropout=0.5):
    with tf.variable_scope(scope_name) as scope:
        x_sat_drop = tf.nn.dropout(sat_local, dropout, name='sat_dropout')
        fc_sat = fc_layer(x_sat_drop, 64 * 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

        scope.reuse_variables()

        x_grd_drop = tf.nn.dropout(grd_local, dropout, name='grd_dropout')
        fc_grd = fc_layer(x_grd_drop, 64 * 512, 4096, 0.005, 0.1, 'fc1', activation_fn=None)
        grd_global = tf.nn.l2_normalize(fc_grd, dim=1)

    return sat_global, grd_global


# =========================================================================================================
# supportive blocks
def fc_layer(x, input_dim, output_dim, init_dev, init_bias, name='fc_layer', activation_fn=tf.nn.relu,
             reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                 trainable=True,
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=init_dev))
        bias = tf.get_variable(name='biases', shape=[output_dim],
                               trainable=True, initializer=tf.constant_initializer(init_bias))

        if activation_fn is not None:
            out = tf.nn.xw_plus_b(x, weight, bias)
            out = activation_fn(out)
        else:
            out = tf.nn.xw_plus_b(x, weight, bias)

    return out


def spatial_aware(input_feature, dimension, trainable, name):
    batch, height, width, channel = input_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_max(input_feature, axis=-1), [-1, height * width])

    with tf.variable_scope(name):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width / 2), dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width / 2), dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1

        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2

        return vec3