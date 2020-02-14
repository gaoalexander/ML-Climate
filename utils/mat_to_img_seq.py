import cv2
import numpy as np

from scipy.io import loadmat
from skimage import data, io
from matplotlib import pyplot as plt


def transformDataTo255(data):
	min_val = np.amin(data)
	data -= min_val
	max_val = np.amax(data)
	data = (data/max_val)*255
	return data.astype(int)

def showPreviewImage(data):
	slice1 = data[:, :, 0]
	io.imshow(slice1)
	plt.show()

def exportImgSequence(data, img_outpath):
	h, w, steps = data.shape
	for i in range(steps):
	    cv2.imwrite(img_outpath%i, data[:, :, i])
	cv2.destroyAllWindows()

def main():
	datapath = "/Volumes/RESEARCH1/CAOS/_data/Training/psiTrain/psiTrain1_30km.mat"
	img_outpath = "/Volumes/RESEARCH1/CAOS/_data/Training/psiTrain/img/psiTrain1_30km.mat/psi1_30km/psi1_30km_%04d.bmp"

	data = loadmat(datapath)['psi1_30km']
	data_img = transformDataTo255(data)
	exportImgSequence(data_img, img_outpath)

if __name__ == "__main__":
	main()