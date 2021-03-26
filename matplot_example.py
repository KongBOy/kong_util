import matplotlib.pyplot as plt
import numpy as np
# fig = plt.figure()
# ### 一種是 xticks 設定 不要有 刻度，但軸線還是會在
# ax1 = fig.add_subplot(2,1,1)
# plt.xticks([])
# plt.yticks([])
# ax1.plot(range(10), 'b-')

# ### 整個軸線都拿掉
# ax2 = fig.add_subplot(2,1,2, sharex=ax1)
# plt.axis('off')
# ax2.plot(range(10), 'r-')

# ### 圖很緊密的靠在一起
# fig.tight_layout()
# plt.show()


x = np.linspace(-3, 3, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = 1 / (1 + np.exp(-x))
y4 = np.exp(x)

fig, ax = plt.subplots(2, 3)

ax[0, 0].plot(x, y1)
ax[0, 1].plot(x, y2)
ax[0, 2].plot(x, y2)
ax[1, 0].plot(x, y3)
ax[1, 1].plot(x, y4)
ax[1, 2].plot(x, y4)

ax[0, 0].set_title("Sine function")
ax[0, 1].set_title("Cosine function")
ax[0, 2].set_title("Cosine function")
ax[1, 0].set_title("Sigmoid function")
ax[1, 1].set_title("Exponential function")
ax[1, 2].set_title("Exponential function")


# fig.tight_layout()  ### 補充一下 tight_layout 加這裡沒有用，因為這裡tight完，下面就又改成用 adjust 的方式排版囉！
plt.subplots_adjust(left=0.01,         ### 左邊界 到 圖的距離 是 幾%
                    bottom=0.01,       ### 下邊界 到 圖的距離 是 幾%
                    right=1.0 - 0.01,  ### 右邊界 到 圖的距離 是 幾%
                    top=1.0 - 0.1,     ### 上邊界 到 圖的距離 是 幾%
                    wspace=0.01,       ### 圖左右之間的 寬
                    hspace=0.01        ### 圖左右之間的 高
                    )   


plt.show()