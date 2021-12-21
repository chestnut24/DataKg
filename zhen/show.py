# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
#
# import h5py as h5
#
# # file_to_read = h5.File('h5test.h5', 'r')
# file_to_read = h5.File('rfi-net.h5', 'r')
# # tod = file_to_read['P/Phase1'].value
# # rfi = file_to_read['RFI/Phase0'].value  # 原h5
# rfi = file_to_read['01/predict'].value
# # rfi = rfi[0:256, 0:128]
# # tod = tod[0:256, 0:128]
# plt.xlabel("Ground Truth")  # 标注
# plt.imshow(np.squeeze(rfi), extent=(640, 768, 990, 1260), cmap="gist_earth", norm=matplotlib.colors.LogNorm())  # 生成像素图片
# plt.show()  # 画图
#
#
#
#
# # plt.imshow(np.squeeze(rfi), extent=(640, 768, 990, 1260), cmap="gist_earth", norm=matplotlib.colors.LogNorm())
# # plt.imshow(np.squeeze(a[0:256, 0:128]), extent=(640, 768, 0, 256), cmap="gist_earth", norm=matplotlib.colors.LogNorm())
#
# # plt.savefig('GroundTruth.eps', format='eps', dpi=300)  # 保存eps图片
# # plt.savefig('GroundTruth.png')  # 保存png文件
# # 不能打开plt.show 否则，生成不了有效的eps文件 注释掉show，即可以正常生成eps文件。
#

# # plt.xlabel("Ground Truth")  # 标注
#
# plt.xlabel("RFI-Gan Label")
# plt.imshow(np.squeeze(rfi), extent=(640, 768, 990, 1260), cmap="gist_earth", )
# plt.savefig('RFI-Gan Label.png')  # 保存png文件
# plt.show()

# # 四个图并列
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
#
# import h5py as h5
#
# file_to_read = h5.File('rfi-gan.h5', 'r')
# rfi = file_to_read['01/predict'].value
#
# file_to_read1 = h5.File('h5test.h5', 'r')
# file_to_read2 = h5.File('rfi-net.h5', 'r')
# file_to_read3 = h5.File('rfi-gan.h5', 'r')
#
# tod1 = file_to_read1['P/Phase1'].value
# rfi2 = file_to_read1['RFI/Phase0'].value  # 原h5
# rfi3 = file_to_read2['01/predict'].value
# rfi4 = file_to_read3['01/predict'].value
# rfi2 = rfi2[0:256, 0:128]
# # rfi = rfi[0:256, 0:128]
# # tod = tod[0:256, 0:128]
#
#
# plt.figure()
# plt.subplots_adjust(left=0.25, bottom=None, right=0.75, top=None, wspace=0.1, hspace=0.4)
# plt.subplot(221)
# plt.xlabel("Mixed Signal")
# plt.imshow(np.squeeze(tod1), extent=(640, 768, 0, 256),cmap="gist_earth", norm=matplotlib.colors.LogNorm())
#
# plt.subplot(222)
# plt.xlabel("Ground Truth")
# plt.imshow(np.squeeze(rfi2), extent=(640, 768, 0, 256),cmap="gist_earth", norm=matplotlib.colors.LogNorm())
#
# plt.subplot(223)
# plt.xlabel("RFI-Gan Label")
# plt.imshow(np.squeeze(rfi4), extent=(640, 768, 0, 256), cmap="gist_earth")
#
# plt.subplot(224)
# plt.xlabel("RFI-Net Label")
# plt.imshow(np.squeeze(rfi3), extent=(640, 768, 0, 256), cmap="gist_earth")
#
# # plt.figure(figsize=(6, 6.5))
# plt.savefig('all.png', dpi=300, bbox_inches='tight')
# # plt.show()





# 三个图数列
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import h5py as h5

# file_to_read1 = h5.File('h5test.h5', 'r')
file_to_read2 = h5.File('result.h5', 'r')
# file_to_read2 = h5.File('rfi-gan2.h5', 'r')
file_to_read1 = h5.File('TEST_MP_PXX_20180530_063000.h5', 'r')

tod1 = file_to_read1['P/Phase1'][()]
rfi2 = file_to_read1['RFI/Phase0'][()]  # 原h5
rfi3 = file_to_read2['01/predict'][()]

tod1 = tod1[0:256, 0:128]
rfi2 = rfi2[0:256, 0:128]



plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.subplot(131)
# plt.subplot(131)
plt.xlabel("Mixed Signal")
plt.imshow(np.squeeze(tod1), extent=(640, 768, 990, 1260),cmap="gist_earth",
                                       norm=matplotlib.colors.LogNorm())

plt.subplot(132)
# plt.subplot(132)
plt.xlabel("Ground Truth")
plt.imshow(np.squeeze(rfi2), extent=(640, 768, 990, 1260),cmap="gist_earth",
                                       norm=matplotlib.colors.LogNorm())

plt.subplot(133)
# plt.subplot(133)
plt.xlabel("SumThreshold Label")
plt.imshow(np.squeeze(rfi3), extent=(640, 768, 990, 1260),cmap="gist_earth")

plt.show()
# plt.savefig('threePic.png')




# # 新 四图 2*2
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
#
# import h5py as h5
#
# #file_to_read = h5.File('rfi-gan.h5', 'r')
# #rfi = file_to_read['01/predict'].value
#
# file_to_read1 = h5.File('h5test.h5', 'r')
# file_to_read2 = h5.File('rfi-net.h5', 'r')
# file_to_read3 = h5.File('rfi-gan.h5', 'r')
#
# tod = file_to_read1['P/Phase1'].value
# rfi = file_to_read1['RFI/Phase0'].value  # 原h5
# rfi3 = file_to_read2['01/predict'].value
# rfi4 = file_to_read3['01/predict'].value
# rfi = rfi[0:256, 0:128]
# # rfi = rfi[0:256, 0:128]
# tod = tod[0:256, 0:128]
#
#
# plt.figure()
# plt.subplots_adjust(left=0.25, bottom=None, right=0.75, top=None, wspace=0.1, hspace=0.4)
# plt.subplot(221)
# plt.xlabel("Mixed Signal")
# plt.imshow(np.squeeze(tod), extent=(640, 768, 990, 1260),cmap="gist_earth", norm=matplotlib.colors.LogNorm())
#
# plt.subplot(222)
# plt.xlabel("Ground Truth")
# plt.imshow(np.squeeze(rfi), extent=(640, 768, 990, 1260),cmap="gist_earth", norm=matplotlib.colors.LogNorm())
#
# plt.subplot(223)
# plt.xlabel("RFI-Gan Label")
# plt.imshow(np.squeeze(rfi4), extent=(640, 768, 990, 1260), cmap="gist_earth")
#
# plt.subplot(224)
# plt.xlabel("RFI-Net Label")
# plt.imshow(np.squeeze(rfi3), extent=(640, 768, 990, 1260), cmap="gist_earth")
#
# # plt.figure(figsize=(6, 6.5))
# plt.savefig('all.png', dpi=300, bbox_inches='tight')
# # plt.show()