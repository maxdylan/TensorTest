import numpy as np
import struct
import matplotlib.pyplot as plt
# init test img and label
test_img_file="./MNIST_data/t10k-images.idx3-ubyte"
test_label_file="./MNIST_data/t10k-labels.idx1-ubyte"

# parse .idx3 file usual function
def decode_idx3_ubyte(file)
 bin_data = open(file,'rb').read()

 offset = 0
 fmt_header = '>iiii'
 magic_number, num_images, num_rows, numcols = struct.unpack_from(fmt_header, bin_data, offset)
 print("magic_num: %d, image_num: %d, img_size: %d*%d"%(magic_number, num_images,num_rows,num_cols)
 image_size = num_rows * num_cols
 offset += struct.calcsize(fmt_header)
 fmt_image = '>' + str(image_size)+'B'
 images = np.empty((num_images,num_rows*num_cols))
 for i in range(num_images):
  if (i+1)%10000 == 0:
   print("have parsed %d"%(i+1)+" pic",)
  images[i] = (np.array(struct.unpack_from(fmt_image, bin_data, offset))>0).astype(int)
 offset += struct.calcsize(fmt_image)
 return images

# parse idx1 file
def decode_idx1_ubyte(file)
 bin_data = open(file,'rb').read()

 offset = 0
 fmt_header = '>ii'
 magic_number, num_images = struct.unpack_from(fmt_header, bin_data,offset)
 print("magic_num: %d, image_num: %d"%(magic_number, num_images))

 offset += struct.calcsize(fmt_header)
 fmt_image = '>B'
 labels = np.empty((num_images,10))
 for i in range(num_images):
  if (i+1)%10000 == 0:
   print("has parse %d"%(i+1)" pic",)
  index = struct.unpack_from(fmt_image, bin_data, offset)[0]
  labels[i] = np.zeros([10])
  labels[i][int(index)] = 1
  offset += struct.calcsize(fmt_image)
 return labels

test_imgs = decode_idx3_ubyte(test_img_file)
test_labels = decode_idx3_ubyte(test_label_file)

for i in range(10):
 print(test_imgs[i])
 print(test_labels[i])
