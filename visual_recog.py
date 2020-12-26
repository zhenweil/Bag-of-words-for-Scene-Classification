import os, math, multiprocessing
from os.path import join
from copy import copy
import numpy as np
from PIL import Image
import visual_words

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    height = wordmap.shape[0]
    width = wordmap.shape[1]
    hist_array = wordmap.reshape(height*width,)
    hist,patches = np.histogram(hist_array, bins = K)
    hist = hist/sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: list of length (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    height = wordmap.shape[0]
    width = wordmap.shape[1]
    hist_SPM = []
    for i in range(L):
        num_row = num_col = pow(2,i)
        len_row = height//num_row
        len_col = width//num_col
        if(i == 0 or i == 1):
            weight = 2**(-i)
        else:
            weight = 2**(i-L-1)
        for r in range(num_row):
            for c in range(num_col):
                wordmap_sub = wordmap[len_row*r:len_row*(r+1),len_col*c:len_col*(c+1)]
                hist_sub = get_feature_from_wordmap(opts, wordmap_sub)*weight
                hist_sub_list = list(hist_sub)
                hist_SPM = hist_SPM + hist_sub_list
    hist_all = hist_SPM/sum(hist_SPM)
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: list of length K
    '''

    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    features = get_feature_from_wordmap_SPM(opts, wordmap)
    return features

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    num_pic = len(train_files)
    features_all = []
    print("Start computing feature pyramids for all images")
    for i in range(num_pic):
        img_path = join(opts.data_dir, train_files[i])
        features = get_image_feature(opts, img_path, dictionary)
        features_all.append(features)
        progress = (i/num_pic) * 100
        if(i % 10 == 0):
            print("progress is: %.2f" % progress, "%.")

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features_all,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num)

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    intersection = np.minimum(histograms, word_hist)
    similarity = np.sum(intersection, axis = 1)
    label_index = np.argmax(similarity)
    return label_index
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    histograms = trained_system['features']
    train_labels = trained_system['labels']
    num_pic = len(test_files)
    conf = np.zeros((8,8))
    accuracy = 0
    print("Start evaluating")
    for i in range(num_pic):
        img_path = join(opts.data_dir, test_files[i])
        word_hist = get_image_feature(opts, img_path, dictionary)
        label_index = distance_to_set(word_hist, histograms)
        label_predict = train_labels[label_index]
        label_true = test_labels[i]
        progress = (i/num_pic) * 100
        conf[label_true, label_predict] += 1
        accuracy = np.trace(conf)/np.sum(conf)
        print("progress is: ", progress, "%. ", "Accuracy is:", int(accuracy*100), "%")
    return conf, accuracy

