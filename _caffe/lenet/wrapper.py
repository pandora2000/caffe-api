#!/usr/bin/env python
"""wrapper.py implements an end-to-end wrapper that classifies an image read
from disk, using the imagenet classifier.
"""

import numpy as np
import os
from skimage import io
from skimage import transform
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filter.rank import enhance_contrast
import caffe

IMAGE_DIM = 28

def prepare_image(filename):
  img = io.imread(filename)
  img_out = rgb2gray(img)
  img_out = transform.resize(img_out, (IMAGE_DIM,IMAGE_DIM))

  img_out = enhance_contrast(img_out, disk(5))

  #binary_image = np.where(img_out > np.mean(img_out),1.0,0.0)
  #io.imsave("images/new.jpg", t(img_out))
  
  t = lambda x : 1.0  - x/255.
  t = np.vectorize(t)
  
  img_out = t(img_out)

  image = np.empty((1, 1, IMAGE_DIM,IMAGE_DIM), dtype=np.float32)

  for i in range(0,IMAGE_DIM):
    for j in range(0,IMAGE_DIM):
      image[:,:,i,j] = img_out[i,j]

  return image


class MNISTClassifier(object):
  """
  The MNISTClassifier is a wrapper class to perform easier deployment
  of models trained on MNIST with Lenet.
  """
  def __init__(self, model_def_file, pretrained_model,
              num_output=10):
    
    self.caffenet = caffe.Net(model_def_file, pretrained_model)
    self._output_blobs = [np.empty((1, num_output, 1, 1), dtype=np.float32)]

  def predict(self, filename):
    input_blob = [prepare_image(filename)]
    self.caffenet.Forward(input_blob, self._output_blobs)
    return self._output_blobs

def main(argv):
  """
  The main function will carry out classification.
  """
  import gflags
  import glob
  import time
  gflags.DEFINE_string("filename", "", "The image filename.")
  gflags.DEFINE_string("root", "", "The folder that contains images.")
  gflags.DEFINE_string("ext", "JPEG", "The image extension.")
  gflags.DEFINE_string("model_def", "", "The model definition file.")
  gflags.DEFINE_string("pretrained_model", "", "The pretrained model.")
  gflags.DEFINE_string("output", "", "The output numpy file.")
  gflags.DEFINE_boolean("gpu", True, "use gpu for computation")
  FLAGS = gflags.FLAGS
  FLAGS(argv)


  prepare_image(FLAGS.filename)

  # net = ImageNetClassifier(FLAGS.model_def, FLAGS.pretrained_model)

  # if FLAGS.gpu:
  #   print 'Use gpu.'
  #   net.caffenet.set_mode_gpu()

  # files = glob.glob(os.path.join(FLAGS.root, "*." + FLAGS.ext))
  # files.sort()
  # print 'A total of %d files' % len(files)
  # output = np.empty((len(files), net._output_blobs[0].shape[1]),
  #                   dtype=np.float32)
  # start = time.time()
  # for i, f in enumerate(files):
  #   output[i] = net.predict(f)
  #   if i % 1000 == 0 and i > 0:
  #     print 'Processed %d files, elapsed %.2f s' % (i, time.time() - start)
  # # Finally, write the results
  # np.save(FLAGS.output, output)
  # print 'Done. Saved to %s.' % FLAGS.output


if __name__ == "__main__":
  import sys
  main(sys.argv)
