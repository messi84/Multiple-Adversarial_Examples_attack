import os
from keras import backend as K
import numpy as np
import random as rd
import tensorflow as tf

from keras.models import load_model

import cv2
import matplotlib.pyplot as plt
from skimage import io
import time
from yolo_model import YOLO



def process_yolo_output(out, anchors, mask):
    """
    Tensor op: Process output features.
    # Arguments
        out - tensor (N, S, S, 3, 4+1+80), output feature map of yolo.
        anchors - List, anchors for box.
        mask - List, mask for anchors.
    # Returns
        boxes - tensor (N, S, S, 3, 4), x,y,w,h for per box.
        box_confidence - tensor (N, S, S, 3, 1), confidence for per box.
        box_class_probs - tensor (N, S, S, 3, 80), class probs for per box.
    """
    batchsize, grid_h, grid_w, num_boxes = map(int, out.shape[0:4])

    box_confidence = tf.sigmoid(out[..., 4:5], name='objectness')  # (N, S, S, 3, 1)
    box_class_probs = tf.sigmoid(out[..., 5:], name='class_probs')  # (N, S, S, 3, 80)

    anchors = np.array([anchors[i] for i in mask]) # Dimension of the used three anchor boxes [[x,x], [x,x], [x,x]].
    # duplicate to shape (batch, height, width, num_anchors, box_params).
    anchors = np.repeat(anchors[np.newaxis, :, :], grid_w, axis=0)          # (S, 3, 2)
    anchors = np.repeat(anchors[np.newaxis, :, :, :], grid_h, axis=0)       # (S, S, 3, 2)
    anchors = np.repeat(anchors[np.newaxis, :, :, :, :], batchsize, axis=0) # (N, S, S, 3, 2)
    anchors_tensors = tf.constant(anchors, dtype=tf.float32, name='anchor_tensors')

    box_xy = tf.sigmoid(out[..., 0:2], name='box_xy') # (N, S, S, 3, 2)
    box_wh = tf.identity(tf.exp(out[..., 2:4]) * anchors_tensors, name='box_wh') # (N, S, S, 3, 2)

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1) #(13, 13, 3, 2)
    grid_batch = np.repeat(grid[np.newaxis, :, :, :, :], batchsize, axis=0)
    box_xy += grid_batch
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)

    # boxes -> (N, S, S, 3, 4)
    boxes = tf.concat([box_xy, box_wh], axis=-1)
    boxes = tf.reshape(boxes, [batchsize, -1, boxes.shape[-2], boxes.shape[-1]], name='boxes') #(N, S*S, 3, 4)
    # box_confidence -> (N, S, S, 3, 1) or 26 or 52
    # box_class_probs -> (N, S, S, 3, 80)
    box_confidence = tf.reshape(box_confidence, [batchsize,
                                                 -1,
                                                 box_confidence.shape[-2],
                                                 box_confidence.shape[-1]], name='box_confidence')
    box_class_probs = tf.reshape(box_class_probs, [batchsize,
                                                   -1,
                                                   box_class_probs.shape[-2],
                                                   box_class_probs.shape[-1]], name='class_probs')
    return boxes, box_confidence, box_class_probs


def process_output(raw_outs):
    """
    Tensor op: Extract b, c, and s from raw outputs.
    # Args:
        raw_outs - Yolo raw output tensor list [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 26, 26, 3, 85)].
    # Returns:
        boxes - Tensors. (N, 3549, 3, 4), classes: (N, 3549, 3, 1), scores: (N, 3549, 3, 80)
    """
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], 
               [30, 61], [62, 45], [59, 119], 
               [116, 90], [156, 198], [373, 326]]
    boxes, objecness, scores = [], [], []

    for out, mask in zip(raw_outs, masks):
        # out -> (N, 13, 13, 3, 85)
        # mask -> one of the masks
        # boxes (N, 13X13, 3, 4), box_confidence (N, 13X13, 3, 1)
        # box_class_probs (13X13, 3, 80) | 26 X 26 |
        b, c, s = process_yolo_output(out, anchors, mask)
        if boxes == []:
            boxes = b
            objecness = c
            scores = s
        else:
            boxes = tf.concat([boxes, b], 1, name='xywh') 
            objecness = tf.concat([objecness, c], 1, name='objectness')
            scores = tf.concat([scores, s], 1, name='class_probs')
    # boxes -> (N, 13*13+26*26+52*52, 3, 4)
    # objecness -> (N, 13*13+26*26+52*52, 3, 1)
    # scores -> (N, 13*13+26*26+52*52, 3, 80)
    return boxes, objecness, scores



class Daedalus:
    """
    Daedalus adversarial example generator based on the Yolo v3 model.
    """
    def __init__(self, sess, model, initial_consts=2000, learning_rate=0.01, 
                 target_class=None, attack_mode='middle', confidence=0.3, binary_search_steps=5,
                 max_iterations=10000):

        # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        self.sess = sess
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.initial_consts = initial_consts
        self.yolo_model = model
        self.confidence = confidence

        def select_class(target_class, boxes, obj_score, class_score, mode):
            
            if False:
                file = '../data/coco_classes.txt'
                with open(file) as f:
                    class_names = f.readlines()
                class_names = [c.strip() for c in class_names]
                like_class = class_names.index(target_class)
            else:
                selected_class = tf.reshape(class_score, [-1, 80])
                like_class = tf.argmax(selected_class[13*13//2], axis = -1)
            
            selected_class = tf.reshape(class_score, [-1, 80])
            selected_boxes = tf.reshape(boxes, [-1, 4])
            selected_object = tf.reshape(obj_score, [-1, 1])
            
            if mode == 'large':
                like_p = np.array([0.8, 0.8, 0.8, 0.8])
            elif mode == 'middle':
                like_p = np.array([0.5, 0.5, 0.5, 0.5])
            elif mode == 'small':
                like_p = np.array([0.3, 0.3, 0.3, 0.3])

            return selected_boxes, selected_object, selected_class, like_class, like_p



        # the perturbation we're going to optimize:
        perturbations = tf.Variable(np.zeros((1, 416, 416, 3)), dtype=tf.float32, name='perturbations')
        # tf variables to sending data to tf:
        self.timgs = tf.Variable(np.zeros((1, 416, 416, 3)), dtype=tf.float32, name='self.timgs')
        self.consts = tf.Variable(np.zeros(1), dtype=tf.float32, name='self.consts')

        # and here's what we use to assign them:
        self.assign_timgs = tf.placeholder(tf.float32, (1, 416, 416, 3))
        self.assign_consts = tf.placeholder(tf.float32, [1])

        # Tensor operation: the resulting image, tanh'd to keep bounded from
        self.newimgs = tf.tanh(perturbations + self.timgs) * 0.5 + 0.5

        # Get prediction from the model:
        outs = self.yolo_model._yolo(self.newimgs)
        # out -> [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 52, 52, 3, 85)]
        boxes, obj_score, class_score = process_output(outs)
        # boxes, obj_score, class_score -> (N, 3549, 3, 4), (N, 3549, 3, 1), (N, 3549, 3, 80)
        boxes, obj_score, class_score, like_class, like_p = select_class(target_class, 
                                                                         boxes, obj_score, class_score, attack_mode)
        # boxes, obj_score, class_score -> (3549*3, 4), (3549*3, 1), (3549*3, 80)

        box_like = boxes[52*52//2]
        obj_like = obj_score[52*52//2]
        cla_like = class_score[52*52//2]
        
        self.bx = boxes[..., 0]
        self.by = boxes[..., 1]
        self.bw = boxes[..., 2]
        self.bh = boxes[..., 3]
        
        slice1 = obj_score[:52*52//2]
        slice2 = obj_score[52*52//2+1:]
        self.otherbox = tf.concat([slice1, slice2], 0)
        
        # Optimisation metrics: 
        self.l2_dist = tf.reduce_sum(tf.square(self.newimgs - (tf.tanh(self.timgs) * 0.5 + 0.5)))
               
        self.loss_otherbox = tf.reduce_sum(tf.square(self.otherbox))
        
        loss_mybox = tf.multiply(obj_like, cla_like[like_class])
        self.loss_mybox = tf.multiply(tf.reduce_sum(tf.square(1-loss_mybox)),100)

        self.mysize = tf.reduce_sum(tf.square(box_like - like_p))

        self.my = self.loss_mybox + self.mysize + self.loss_otherbox
        self.loss_f = self.consts * self.my
        self.loss = self.loss_f + self.l2_dist

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE,
                                           global_step=global_step,
                                           decay_steps=400,
                                           decay_rate=0.5)
        self.add_global = global_step.assign_add(1)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer_mybox = tf.train.AdamOptimizer(learning_rate)
        
        self.train = optimizer.minimize(self.loss, var_list=[perturbations])
        self.train_mybox = optimizer_mybox.minimize(self.loss_mybox, var_list=[perturbations])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # self.setup: initialization of hyper-pramater c and input images
        # self.init: initialization of perturbation and other variables we created
        self.setup = []
        self.setup.append(self.timgs.assign(self.assign_timgs))
        self.setup.append(self.consts.assign(self.assign_consts))
        self.init = tf.variables_initializer(var_list=[perturbations] + new_vars)

        
    def attack(self, img):
        """
        Run the attack on a batch of images and labels.
        """
        def check_success(loss, init_loss):
            return loss <= init_loss * (1 - self.confidence)

        # convert images to arctanh-space
        img = np.arctanh((img - 0.5) / 0.5)

        # set the lower and upper bounds of the constsant.
        lower_bound = 1
        consts = self.initial_consts
        upper_bound = 1e9

        # store the best l2, score, and image attack
        best_l2 = 1e10
        o_bestloss = 1e10
        best_pert = np.zeros(img.shape)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)

            # The last iteration (if we run many steps) repeat the search once.
            if outer_step == self.BINARY_SEARCH_STEPS - 1:
                consts = upper_bound

            # set the variables so that we don't have to send them over again.
            self.sess.run(self.setup, {self.assign_timgs: img[np.newaxis,:,:,:],
                                       self.assign_consts: np.full(1, consts)})

            # start gradient descent attack
            print '\nparameter C is ', consts
            init_loss = self.sess.run(self.loss)
            init_adv_losses = self.sess.run(self.loss_f)
            prev = init_loss * 1.1
            
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack on a single example
                if iteration % 10 == 0:
                    _ = self.sess.run(self.train_mybox)
                    
                _, loss, l2_dist, loss_f, out_img, loss_mybox = self.sess.run([self.train, self.loss, 
                                                                   self.l2_dist, self.my, self.newimgs, self.loss_mybox])
                
                _ = self.sess.run(self.add_global)
                
                #loss_otherbox, loss_mybox, mysize, my, loss_f = self.sess.run([self.loss_otherbox,
                #                                                               self.loss_mybox,
                #                                                               self.mysize,
                #                                                               self.my,
                #                                                               self.loss_f])
                #print(round(loss_otherbox,3), round(loss_mybox,3), round(mysize,3), round(my,3), round(loss_f,3))
                
                #a, b = self.sess.run([self.obj_like, self.cla])
                #return a, b
                
                if (iteration+1) % 100 == 0:
                    print '\n===========iteration:', iteration+1, '============'
                    print 'total losses: ', round(loss, 3)
                    print 'l2_dist losses: ', round(l2_dist, 3)
                    print 'func losses: ', round(loss_f, 3)
                    print 'loss_mybox losses: ', round(loss_mybox, 3)
                    
                # check if we should abort search if we're getting nowhere.
                if (iteration+1) % 1000 == 0:    
                    if loss > prev:
                        break
                    prev = loss
                
                # update the best result found so far
                if l2_dist < best_l2*1.1 and check_success(loss_f, init_adv_losses):
                    best_l2 = l2_dist
                    o_bestloss = loss_f
                    best_pert = out_img[0]

            # adjust the constsant as needed
            if check_success(loss_f, init_adv_losses):
                # success, divide consts by two
                upper_bound = min(upper_bound, consts)
                if upper_bound < 1e8:
                    consts = (lower_bound + upper_bound) / 2
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                lower_bound = max(lower_bound, consts)
                if upper_bound < 1e8:
                    consts = (lower_bound + upper_bound) / 2
                else:
                    consts *= 10

        return np.array(best_pert), np.array(best_l2)

