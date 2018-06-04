import cv2

def gaussian(image, numlevels):
	"""Constructs gaussian pyramid

	Arguments:
		image : Input image (monochrome or color)
		numlevels : Number of levels to compute

	Return:
		List of progressively smaller (i.e. lower frequency) images
	"""

	pyramid = [ image ]
	for level in range(numlevels):
		image = cv2.pyrDown(image)
		pyramid.append(image)

	return pyramid

def laplacian(image, numlevels):
	"""Constructs laplacian pyramid

	Arguments:
		image : Input image (monochrome or color)
		numlevels : Number of levels to compute

	Return:
		List of progressively smaller (i.e. lower frequency) images
	"""
	if (3 == image.ndim):
		isColor = True

	# compute gaussian pyramid
	gausianPyramid = gaussian(image, numlevels)

	# empty laplacian pyramid
	laplacianPyramid = []

	# compute laplacian layers for 0 <= i <= N: L_i = g_i - expand(g_{i+1})
	for i in range(len(gausianPyramid)-1):
		a = gausianPyramid[i]
		b = cv2.pyrUp(gausianPyramid[i+1])

		if isColor:
			a_width, a_height, _ = a.shape
			b_width, b_height, _ = b.shape
		else:
			a_width, a_height = a.shape
			b_width, b_height = b.shape

		# compute least common shape
		w = min(a_width, b_width)
		h = min(a_height, b_height)

		# take difference of common area (uses saturation arithmetic)
		if isColor:
			laplacianPyramid.append( cv2.subtract(a[0:w,0:h,:], b[0:w,0:h,:]) )
		else:
			laplacianPyramid.append( cv2.subtract(a[0:w,0:h], b[0:w,0:h]) )

	# final laplacian layer is copy of gaussian final layer: L_N = g_N
	laplacianPyramid.append( gausianPyramid[-1] )

	return laplacianPyramid