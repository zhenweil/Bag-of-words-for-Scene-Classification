import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import random
from sklearn.cluster import KMeans

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # If img is gray scale, convert it to 3d
    if(img.ndim == 2):
        img = np.dstack((img,img,img))
        print("grey img")
    img_lab = skimage.color.rgb2lab(img)
    scales = opts.filter_scales
    height = img.shape[0]
    width = img.shape[1]
    num_scales = len(scales)
    layers = num_scales*3
    filter_responses = np.empty([height, width, layers*4])
    for i in range(num_scales):
        for j in range(3):
            rsp_gauss = scipy.ndimage.gaussian_filter(img_lab[:,:,j], scales[i])
            filter_responses[:,:,3*i+j] = rsp_gauss
    for i in range(num_scales):
        for j in range(3):
            rsp_laplace = scipy.ndimage.gaussian_laplace(img_lab[:,:,j],scales[i])
            filter_responses[:,:,layers+3*i+j] = rsp_laplace
    for i in range(num_scales):
        for j in range(3):
            rsp_gaussdx = scipy.ndimage.gaussian_filter(img_lab[:,:,j],scales[i],[0,1])
            filter_responses[:,:,2*layers+3*i+j] = rsp_gaussdx
    for i in range(num_scales):
        for j in range(3):
            rsp_gaussdy = scipy.ndimage.gaussian_filter(img_lab[:,:,j], scales[i],[1,0])
            filter_responses[:,:,3*layers+3*i+j] = rsp_gaussdy
    return filter_responses

def compute_dictionary_one_image(opts, path_name):
    #def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    [input]
    * opts    : options
    * path_name    : partial name of the picture which does not include data_dir
    [output]
    * filter_responses: numpy.ndarray of shape (H*W,3F)
    [comment]
    * path of data_dir is not needed here because data_dir will be joined with picture path below
    '''
    num_sample = opts.alpha
    img_path = join(opts.data_dir, path_name)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = extract_filter_responses(opts, img)
    height = filter_responses.shape[0]
    width = filter_responses.shape[1]
    depth = filter_responses.shape[2]
    random_index = random.sample(range(0, height*width), num_sample)
    filter_responses_reshaped = filter_responses.reshape((height*width, depth))
    dict_one_image = filter_responses_reshaped[random_index,:]
    return dict_one_image

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    filter_scales = opts.filter_scales
    num_scales = len(filter_scales)
    num_layers = 12*num_scales
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    words = []
    num_pic = len(train_files)
    print("started building dictionary")
    print("Number of training images is: ", num_pic)

    num_sample = opts.alpha
    '''pool = multiprocessing.Pool(processes = n_worker)
    args = [(opts, path_name) for path_name in train_files[0:10]]
    #pool.map(compute_dictionary_one_image, args)'''
    for i in range(num_pic):
        path_name = train_files[i]
        dict_one_image = compute_dictionary_one_image(opts, path_name)
        dict_one_image_list = list(dict_one_image)
        words.append(dict_one_image_list)
        progress = (i/num_pic) * 100
        if(i % 10 == 0):
            print("progress is: %.2f" % progress, "%.")
    words = np.asarray(words).reshape(num_pic*num_sample,num_layers)
    kmeans = KMeans(n_clusters = K).fit(words)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir,'words.npy'), words)
    np.save(join(out_dir,'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    K = opts.K
    filter_scales = opts.filter_scales
    num_scales = len(filter_scales)
    num_layers = 12*num_scales
    filter_responses = extract_filter_responses(opts, img)
    height = filter_responses.shape[0]
    width = filter_responses.shape[1]
    filter_responses_reshaped = filter_responses.reshape(height*width,num_layers)
    dist = scipy.spatial.distance.cdist(filter_responses_reshaped, dictionary, metric = 'euclidean')
    
    visual_word = []
    num_pixel = dist.shape[0]
    for i in range(num_pixel):
        word = np.argmin(dist[i,:])
        visual_word.append(word)
    visual_word = np.asarray(visual_word)
    wordmap = visual_word.reshape(height, width)
    return wordmap

