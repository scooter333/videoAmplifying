#!/usr/bin/env python

"""
Eulerian Video Magnification (EVM) Demo
"""

import time
import sys
import cv2
import numpy as np


def band_pass_filter(input, wl, wh, sample_rate):
	"""Applies ideal band-pass filter to a given video

	Arguments:
		input : video to be filtered (as a 4-d numpy array (time, height,
		        width, channels))
		wl : lower cut-off frequency of band-pass filter
		wh : upper cut-off frequency of band-pass filter
		sample_rate : 

	Return:
		Temporally filtered video as 4-d array
	"""
	num_frames, height, width, num_channels = input.shape

	# construct 1-d frequency domain mask
	freq = range(num_frames)
	freq = map(lambda x: float(x) / num_frames * sample_rate, freq)
	mask = np.asarray(map(lambda x: x > wl and x < wh, freq))

	# copy mask such that it applies to each pixel and channel
	#
	# important: numpy prepends new axes by default whereas matlab postpends
	#            hence, we use np.newaxis to explicitly define postpending.
	mask = np.tile(mask[:,np.newaxis,np.newaxis,np.newaxis], \
	               (1, height, width, num_channels))

	# transform input frames into frequency domain
	F = np.fft.fft(input, axis=0)

	# suppress out-of-band frequencies
	F[np.invert(mask)] = 0

	# transform back into time domain
	return np.real(np.fft.ifft(F, axis=0))


def processImg(filename, numlevels=4, alpha=50., chromAttenuation=1.):
		"""Process video

		Arguments:
			numlevels : Number of pyramid levels to compute
		"""
	#	from cv import CV_32F
	#	from cv import CV_BGR2YCrCb, CV_YCrCb2BGR
		from pyramid import gaussian

		video = cv2.VideoCapture(filename)

		frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
		frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		numChannels = 3

		# create windows for original and processed video
		cv2.namedWindow("original")
		cv2.namedWindow("processed")

		# position displayed windows
		cv2.moveWindow("original", 100, 100)
		cv2.moveWindow("processed", 100 + frameWidth + 50, 100)

		# allocate memory for input frames
		in_frames = np.ndarray(shape=(frameCount, \
		                                   frameHeight, \
		                                   frameWidth, \
		                                   numChannels), \
		                            dtype=np.float32)

		for frameNumber in range(frameCount):

			# read frame of video
			(retval, frame) = video.read()
			if not retval:
				break

			# convert to floating-point (no double in opencv)
			frame = np.asarray(frame, dtype=np.float32) / 255.

			# separate luminance and chrominance
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

			# store frame into memory
			in_frames[frameNumber] = frame

		# compute pyramid on first frame
		pyramid = gaussian(in_frames[0], numlevels)
		height, width, _ = pyramid[-1].shape

		# allocate memory for downsampled frames
		ds_frames = np.ndarray(shape=(frameCount, \
		                                   height, \
		                                   width, \
		                                   numChannels), \
		                            dtype=np.float32)
		ds_frames[0] = pyramid[-1]
		                                      
		for frameNumber in range(1, frameCount):

			# spatial decomposition (specify laplacian or gaussian)
			pyramid = gaussian(in_frames[frameNumber], numlevels)

			# store downsampled frame into memory
			ds_frames[frameNumber] = pyramid[-1]
		output = band_pass_filter(ds_frames, 50./60., 60./60., 30)
		output[:,:,:,0] *= alpha
		output[:,:,:,1] *= (alpha * chromAttenuation)
		output[:,:,:,2] *= (alpha * chromAttenuation)
		count = 1
	
		for i in range(frameCount):
			#from cv import CV_INTER_CUBIC

			orig = in_frames[i]

			# enlarge to match size of original frame (keep as 32-bit float)
			filt = cv2.resize(output[i], (frameWidth, frameHeight), \
			                  interpolation=cv2.INTER_CUBIC)
			filt = filt.astype(np.float32)

			filt = filt + orig

			filt = cv2.cvtColor(filt, cv2.COLOR_YCrCb2BGR)

			filt[filt > 1] = 1
			filt[filt < 0] = 0
#			filt = cv2.add(orig, filt, dtype=CV_32F)

			# preview
			cv2.imshow('original', cv2.cvtColor(orig, cv2.COLOR_YCrCb2BGR))
			cv2.imshow('processed', filt)


			#cv2.imwrite('original'+str(count)+'.png',cv2.cvtColor(orig, cv2.COLOR_YCrCb2BGR)*256)
			#cv2.imwrite('filt'+str(count)+'.png',filt*256)
	
			count+=1
			cv2.waitKey(20)
processImg("evmtest.mp4")
#fileName = input("Enter video file name in quotes.")
#processImg(fileName) 