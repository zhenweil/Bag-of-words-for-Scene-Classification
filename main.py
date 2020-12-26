from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

def main():
    opts = get_opts()
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

if __name__ == '__main__':
   main()
