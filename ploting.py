import numpy
import matplotlib.pyplot as plt
with open('mnist_test_10.csv') as f:
	n=f.readlines()
a=n[8].split(',')
image_array=numpy.asarray(a[1:],dtype=float)
image_array=image_array.reshape(28,28)
plt.imshow(image_array,cmap='Greys',interpolation='None')
plt.show()

a=n[9].split(',')
image_array=numpy.asarray(a[1:],dtype=float)
image_array=image_array.reshape(28,28)
plt.imshow(image_array,cmap='Greys',interpolation='None')
plt.show()
