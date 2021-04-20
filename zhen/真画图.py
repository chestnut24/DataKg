def show_result_image():
	# import cv2
	import matplotlib
	f1 = h5.File('collected_hfd5.h5', 'r')
	f2 = h5.File(os.path.join(TEST_RESULT_DIRECTORY, 'Score.h5'), 'r')
	f3 = h5.File('../data_set/sum_threshold.h5', 'r')
	f4 = h5.File('../data_set/knn.h5', 'r')
	f5 = h5.File('../data_set/test_result/unet_test_result/Score.h5', 'r')
	i = 3
	rfi_mask = f1['%04d/rfi_mask' % (i + 2900)].value * 200
	rfi = f1['%04d/rfi' % (i + 2900)].value
	tod = f1['%04d/tod' % (i + 2900)].value
	rfi_net = f2['%02d/predict' % i].value * 200
	sum_threshold = f3['%02d/predict' % i].value * 200
	knn = f4['%02d/prediction' % i].value * 200
	unet = f5['%02d/predict' % i].value * 200
	# cv2.imshow('ground_truth', ground_truth)
	# cv2.imshow('prediction', predition)
	# cv2.waitKey(0)

	plt.figure()
	plt.subplots_adjust(hspace=0.4, wspace=0.4)

	# 显示为6个图片
	plt.subplot(231)
	plt.title(s='Data', fontsize=35)
	plt.imshow(tod, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth",  norm=matplotlib.colors.LogNorm())

	plt.subplot(232)
	plt.title(s='RFI', fontsize=35)
	plt.imshow(rfi, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth",  norm=matplotlib.colors.LogNorm())

	plt.subplot(233)
	plt.title(s='Ground Truth', fontsize=35)
	plt.imshow(rfi_mask, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth")

	plt.subplot(234)
	plt.title(s='Sum_Threshold', fontsize=35)
	plt.imshow(sum_threshold, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth")

	plt.subplot(235)
	plt.title(s='U-Net', fontsize=35)
	plt.imshow(unet, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth")

	plt.subplot(236)
	plt.title(s='Our Work', fontsize=35)
	plt.imshow(rfi_net, aspect="auto",
	           extent=(0, 800, 990, 1260),
	           cmap="gist_earth")

	# 显示为4个图片
	# plt.subplot(221)
	# plt.title(s='RFI', fontsize=20)
	# # plt.xlabel()
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.imshow(rfi, aspect="auto",
	#            extent=(0, 800, 990, 1260),
	#            cmap="gist_earth", norm=matplotlib.colors.LogNorm())
	#
	# plt.subplot(222)
	# plt.title(s='Ground_truth', fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.imshow(rfi_mask, aspect="auto",
	#            extent=(0, 800, 990, 1260),
	#            cmap="gist_earth")
	#
	# plt.subplot(223)
	# plt.title(s='Sum_Threshold', fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.imshow(sum_threshold, aspect="auto",
	#            extent=(0, 800, 990, 1260),
	#            cmap="gist_earth")
	#
	# plt.subplot(224)
	# plt.title(s='Our Work', fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.imshow(rfi_net, aspect="auto",
	#            extent=(0, 800, 990, 1260),
	#            cmap="gist_earth")
	plt.tick_params(labelsize=25)
	plt.show()

	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()