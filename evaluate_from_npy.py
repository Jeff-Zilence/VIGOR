
import os,sys
from models import SAFA, SAFA_delta_validate, clean_siamese_delta_validate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataloader import DataLoader
import random


# delta generation and RGradCAM visualization from https://github.com/Jeff-Zilence/Explain_Metric_Learning
class ExplanationGenerator:

    def __init__(self, mode='CVUSA',delta=False,path=None):
        self.mode = mode
        self.size_sat = [320, 320]
        self.size_grd = [320, 640]
        self.ori_size_sat = [640,640]
        self.ori_size_grd = [1024,2048]
        self.load_model(delta=delta,path=path)
        self.Decomposition = None

    # read image, subtract bias, convert to rgb for imshow
    def read(self, path, size):
        image = cv2.imread(path).astype(np.float)
        image = cv2.resize(image,(size[1],size[0]))
        image_show = image[:,:,::-1]/255.
        image = image - np.array([103.939, 116.779, 123.6])
        return np.expand_dims(image,axis=0), image_show

    def get_input_from_path(self, path_1, path_2):
        '''
            load two images from paths
            sat denotes satallite (Aerial view) and grd denotes ground (Street view)
        '''
        inputs_sat, image_sat = self.read(path_1, size=self.size_sat)
        inputs_grd, image_grd = self.read(path_2, size=self.size_grd)

        return inputs_sat, image_sat, inputs_grd, image_grd

    def load_model(self, path = None, delta=False):
        '''
            Load the trained model, you may change the path of model here.
            Get the cosine similarity and parameters of fc layer
        '''
        self.sat_x = tf.placeholder(tf.float32, [None, self.size_sat[0], self.size_sat[1], 3], name='sat_x')
        self.grd_x = tf.placeholder(tf.float32, [None, self.size_grd[0], self.size_grd[1], 3], name='grd_x')
        self.keep_prob = tf.placeholder(tf.float32)
        self.index = tf.placeholder(tf.float32, [4096], name='grd_x')
        if delta:
            if 'clean' in self.mode:
                self.sat_global, self.grd_global, self.sat_local, self.grd_local, self.delta = clean_siamese_delta_validate(
                    self.sat_x, self.grd_x, out_dim=100 if 'classification' in self.mode else 2)
            else:
                self.sat_global, self.grd_global, self.sat_local, self.grd_local, self.delta = SAFA_delta_validate(
                self.sat_x, self.grd_x, out_dim=100 if 'classification' in self.mode else 2)
        else:
            self.sat_global, self.grd_global, self.fc_sat, self.fc_grd, self.sat_local, self.grd_local = SAFA(self.sat_x, self.grd_x, original=True)
            self.product = tf.reduce_sum(self.sat_global*self.grd_global,axis=1)
            self.product_ori = tf.reduce_sum(self.fc_sat*self.fc_grd,axis=1)
            self.product_index = tf.reduce_sum(self.fc_sat*self.fc_grd*self.index,axis=1)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

        print('load model...')
        saver.restore(self.sess, path)
        print("Model loaded from: %s" % path)

    def imshow_convert(self, raw):
        '''
            convert the heatmap for imshow
        '''
        heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
        return heatmap

    def GradCAM(self, sess, cost , target, feed_dict, size):
        gradient = tf.gradients(cost, target)[0]
        conv_output, conv_first_grad = sess.run([target, gradient], feed_dict=feed_dict)

        # compute the average value
        weights = np.mean(conv_first_grad[0], axis = (0, 1))
        grad_CAM_map = np.sum(weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1],size[0]))
        return cam

    def RGradCAM(self, sess, cost , target, feed_dict, size):
        # rectified Grad-CAM, a variant
        gradient = tf.gradients(cost, target)[0]
        conv_output, conv_first_grad = sess.run([target, gradient], feed_dict=feed_dict)

        # remove the heuristic GAP step
        weights = conv_first_grad[0]
        grad_CAM_map = np.sum(weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = cv2.resize(cam, (size[1], size[0]))
        return cam

    def visualize(self, path_1, path_2, show = True,delta=None):
        inputs_sat, image_1, inputs_grd, image_2 = self.get_input_from_path(path_1=path_1, path_2=path_2)
        feed_dict = {self.sat_x: inputs_sat, self.grd_x: inputs_grd}
        sat_global_val, grd_global_val = self.sess.run([self.sat_global,self.grd_global], feed_dict=feed_dict)
        product_vector = sat_global_val * grd_global_val
        product_order = np.argsort(product_vector[0])

        # Grad-CAM
        gradcam_1 = self.GradCAM(sess=self.sess, cost=self.product_ori, target=self.sat_local, feed_dict=feed_dict,
                                 size=self.size_sat)
        gradcam_2 = self.GradCAM(sess=self.sess, cost=self.product_ori, target=self.grd_local, feed_dict=feed_dict,
                                 size=self.size_grd)
        if show:
            image_overlay_1 = image_1 * 0.7 + self.imshow_convert(gradcam_1) / 255.0 * 0.3
            image_overlay_2 = image_2 * 0.7 + self.imshow_convert(gradcam_2) / 255.0 * 0.3

            plt.figure()
            plt.suptitle('Grad-CAM')
            plt.subplot(2, 2, 1)
            # plt.imshow(self.imshow_convert(gradcam_1))
            plt.imshow(gradcam_1)
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(gradcam_2))
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.resize(image_overlay_1,(self.ori_size_sat[1],self.ori_size_sat[0])))
            if delta is not None:
                plt.plot(320.-delta[1], 320.+delta[0],'ro',markersize = 7)
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.resize(image_overlay_2,(self.ori_size_grd[1],self.ori_size_grd[0])))

        # RGrad-CAM
        rgradcam_1 = self.RGradCAM(sess=self.sess, cost=self.product_ori, target=self.sat_local, feed_dict=feed_dict,
                                 size=self.size_sat)
        rgradcam_2 = self.RGradCAM(sess=self.sess, cost=self.product_ori, target=self.grd_local, feed_dict=feed_dict,
                                 size=self.size_grd)
        if show:
            image_overlay_1 = image_1 * 0.7 + self.imshow_convert(rgradcam_1) / 255.0 * 0.3
            image_overlay_2 = image_2 * 0.7 + self.imshow_convert(rgradcam_2) / 255.0 * 0.3

            plt.figure()
            plt.suptitle('RGrad-CAM')
            plt.subplot(2, 2, 1)
            plt.imshow(self.imshow_convert(rgradcam_1))
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(rgradcam_2))
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.resize(image_overlay_1, (self.ori_size_sat[1],self.ori_size_sat[0])))
            if delta is not None:
                plt.plot(320.-delta[1], 320.+delta[0],'ro',markersize = 7)
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.resize(image_overlay_2, (self.ori_size_grd[1],self.ori_size_grd[0])))
        return [gradcam_1,gradcam_2],[rgradcam_1,rgradcam_2]

    def get_delta(self, data_loader, order_list,name='delta_freeze.npy'):
        delta_list = []
        while True:
            batch_sat, batch_grd, batch_list = data_loader.next_pair_scan_order(20,order_list)
            if batch_sat is None:
                break
            if batch_list[0] % 100 == 0:
                print(batch_list[0])
            feed_dict = {self.sat_x: batch_sat, self.grd_x: batch_grd}
            delta_val = self.sess.run(self.delta, feed_dict=feed_dict)
            delta_list.extend(delta_val)
        delta_list = np.array(delta_list)
        delta_list = delta_list *320.
        np.save(name,delta_list)
        return delta_list


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C = np.sin(lat_A)*np.sin(lat_B) + np.cos(lat_A)*np.cos(lat_B)*np.cos(lng_A-lng_B)
    distance = R*np.arccos(C)
    return distance


# compute the delta unit for each reference location [Lat, Lng], 320 is half of the image width
# 0.114 is resolution in meter
# reverse equation from gps2distance: https://en.wikipedia.org/wiki/Great-circle_distance
def Lat_Lng(Lat_A, Lng_A, distance=[320*0.114, 320*0.114]):
    if distance[0] == 0 and distance[1] == 0:
        return np.zeros(2)

    lat_A = Lat_A * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    R = 6371004.
    C_lat = np.cos(distance[0]/R)
    C_lng = np.cos(distance[1]/R)
    delta_lat = np.arccos(C_lat)
    delta_lng = np.arccos((C_lng-np.sin(lat_A)*np.sin(lat_A))/np.cos(lat_A)/np.cos(lat_A))
    return np.array([delta_lat * 180. / np.pi, delta_lng * 180. / np.pi])


def generate_distances(data_loader, order_list, delta_list):
    distance_list = []
    for i in range(data_loader.test_data_size):
        truth_gps = np.array(data_loader.test_list[i].split(',')[1:3]).astype(np.float)

        prediction_name = data_loader.test_sat_list[order_list[i]]
        prediction_gps_center = np.array(prediction_name.replace('.png','').split('_')[1:3]).astype(np.float)
        prediction_gps_delta = - Lat_Lng(prediction_gps_center[0],prediction_gps_center[1]) * delta_list[i] / 320.
        prediction_gps = prediction_gps_center + prediction_gps_delta

        distance = gps2distance(truth_gps[0],truth_gps[1],prediction_gps[0],prediction_gps[1])
        distance_list.append(distance)
    return np.array(distance_list)


def distance_accuracy(distance_list):
    accuracy_list = []
    threshold_list = []
    for i in range(1,500):
        threshold = 1.1**(i/5.)
        accuracy_list.append(np.sum((distance_list< threshold)*1.)/distance_list.shape[0])
        threshold_list.append(threshold)
    return np.array(threshold_list), np.array(accuracy_list)


# for simulation of noisy GPS
def generate_noisy_dict(data_loader):
    noisy_dict = {}
    search_distance = 200
    sat_lat_list = []
    sat_lng_list = []
    distance_list = []
    for j, sat in enumerate(data_loader.test_sat_list):
        sat_lat_list.append(np.array(sat.replace('.png', '').split('_')[-2]).astype(np.float))
        sat_lng_list.append(np.array(sat.replace('.png', '').split('_')[-1]).astype(np.float))
    sat_lat_list = np.array(sat_lat_list)
    sat_lng_list = np.array(sat_lng_list)

    for i in range(data_loader.test_data_size):
        if i % 1000 == 0:
            print(i)
        name = data_loader.test_list[i]
        lat = np.array(name.split('/')[-1].split(',')[-3]).astype(np.float)
        lng = np.array(name.split('/')[-1].split(',')[-2]).astype(np.float)
        noisy_lat = lat + random.uniform(-1, 1) * data_loader.delta_unit[0] * 100 / (320 * 0.114)
        noisy_lng = lng + random.uniform(-1, 1) * data_loader.delta_unit[1] * 100 / (320 * 0.114)
        lat_lower = noisy_lat - data_loader.delta_unit[0] * search_distance / (320 * 0.114)
        lat_upper = noisy_lat + data_loader.delta_unit[0] * search_distance / (320 * 0.114)
        lng_lower = noisy_lng - data_loader.delta_unit[1] * search_distance / (320 * 0.114)
        lng_upper = noisy_lng + data_loader.delta_unit[1] * search_distance / (320 * 0.114)
        lat_logit = (sat_lat_list > lat_lower) * (sat_lat_list < lat_upper)
        lng_logit = (sat_lng_list > lng_lower) * (sat_lng_list < lng_upper)
        where = np.where(lat_logit * lng_logit)
        noisy_dict[i] = where[0]
        distance = gps2distance(lat, lng, noisy_lat, noisy_lng)
        distance_list.append(distance)
    # np.save('./data/noisy_dict_cross_200.npy', noisy_dict)
    # np.save('./data/noisy_distance_cross.npy',np.array(distance_list))

    return noisy_dict


def meter_level_localization_from_npy(load_path=None,show=True):
    data_loader = DataLoader(same_area=True)
    # =======================================================================================
    load_path = './data/same_area/Overall'

    # uncomment this if you want to generate delta from checkpoints
    # generator = ExplanationGenerator(delta=True, path='model path')

    sat_descriptor = np.load(os.path.join(load_path, 'sat_global_descriptor.npy'))
    grd_descriptor = np.load(os.path.join(load_path, 'grd_global_descriptor.npy'))
    similarity = np.matmul(grd_descriptor, np.transpose(sat_descriptor))
    order_list = np.argmax(similarity, axis=1)
    del(similarity)

    # uncomment this if you want to generate delta from checkpoints, prediction order list is needed to generate delta
    # delta_list = generator.get_delta(data_loader,order_list,name='./data/delta_same_regression.npy')

    delta_list = np.load('./data/same_area/Overall/delta_same_regression.npy')
    distance_list = generate_distances(data_loader, order_list, delta_list)
    threshold_list, accuracy_list = distance_accuracy(distance_list)

    # =======================================================================================
    # delta = 0 when there is no offset prediction, always use the center as the predicted location
    delta_list_center = np.zeros([data_loader.test_data_size, 2])

    load_path = './data/same_area/SAFA_Mining'
    sat_descriptor = np.load(os.path.join(load_path, 'sat_global_descriptor.npy'))
    grd_descriptor = np.load(os.path.join(load_path, 'grd_global_descriptor.npy'))
    similarity = np.matmul(grd_descriptor, np.transpose(sat_descriptor))
    order_list = np.argmax(similarity, axis=1)
    del(similarity)

    distance_list = generate_distances(data_loader, order_list, delta_list_center)
    threshold_list, accuracy_list_binary = distance_accuracy(distance_list)
    # =======================================================================================
    load_path = './data/same_area/SAFA'
    sat_descriptor = np.load(os.path.join(load_path, 'sat_global_descriptor.npy'))
    grd_descriptor = np.load(os.path.join(load_path, 'grd_global_descriptor.npy'))
    similarity = np.matmul(grd_descriptor, np.transpose(sat_descriptor))
    order_list = np.argmax(similarity, axis=1)
    del(similarity)

    distance_list = generate_distances(data_loader, order_list, delta_list_center)
    threshold_list, accuracy_list_safa = distance_accuracy(distance_list)
    # =======================================================================================
    load_path = './data/same_area/Siamese_VGG'
    sat_descriptor = np.load(os.path.join(load_path, 'sat_global_descriptor.npy'))
    grd_descriptor = np.load(os.path.join(load_path, 'grd_global_descriptor.npy'))
    similarity = np.matmul(grd_descriptor, np.transpose(sat_descriptor))
    order_list = np.argmax(similarity, axis=1)
    del(similarity)

    distance_list = generate_distances(data_loader, order_list, delta_list_center)
    threshold_list, accuracy_list_siamese = distance_accuracy(distance_list)
    # =======================================================================================
    if show:
        n = np.argmin(np.abs(threshold_list-10))

        plt.plot(threshold_list, accuracy_list_siamese * 100., '-c')
        plt.plot(threshold_list, accuracy_list_safa * 100., '-m')
        plt.plot(threshold_list, accuracy_list_binary * 100., '-g')
        plt.plot(threshold_list, accuracy_list * 100., '-r')

        plt.legend(['Siamese-VGG','SAFA','SAFA+Mining','Ours'],loc=2,fontsize=12)
        plt.ylabel('Accuracy (%)', FontSize = 16)
        plt.xlabel('Threshold (m)', FontSize = 16)
        plt.axis([1,40,0,55])
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.show()


# Evaluate based on npy files, the full validation from model takes a long time and requires large memory
# the code of delta.npy generation is provided in the ExplanationGenerator
# feature generation is included in train_SAFA.py
if __name__=='__main__':
    meter_level_localization_from_npy()
