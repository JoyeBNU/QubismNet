import numpy as np
from PIL import Image


def state2image(state, d, is_rescale=False):
    num = int(round(np.log(state.size) / np.log(d)))
    if state.ndim != num:
        state = state.reshape(np.ones((num, ), dtype=int) * d)
    num_h = int(round(num/2))
    image = np.zeros((d**num_h, d**num_h))
    config = [0] * (num + 1)
    while config[0] == 0:
        x = list()
        for nx in range(0, num+1, 2):
            x.append(config[nx])
        y = list()
        for ny in range(1, num+1, 2):
            y.append(config[ny])
        x = list2num(x, d)
        y = list2num(y, d)
        ind = list2index(config[1:])
        image[x, y] = eval('state[' + ind + ']')
        config[-1] += 1
        for n in range(num, 0, -1):
            if config[n] == d:
                config[n] = 0
                config[n-1] += 1
            else:
                break
    if is_rescale:
        image = image / max(abs(image.reshape(-1, ))) * 255
    return image


def state2image_2(state, d, is_rescale=False):
    # s=(X_1, X_2,...,X_n, Y_1, Y_2,...,Y_n)
    # 即前一半的qubits决定x轴，后一半的qubits决定y轴
    num = int(round(np.log(state.size) / np.log(d)))
    if state.ndim != num:
        state = state.reshape(np.ones((num, ), dtype=int) * d)
    num_h = int(round(num/2))
    image = np.zeros((d**num_h, d**num_h))
    config = [0] * (num + 1)
    while config[0] == 0:
        x = list()
        for nx in range(0, num_h+1, 1):
            x.append(config[nx])
        y = list()
        for ny in range(num_h+1, num+1, 1):
            y.append(config[ny])
        x = list2num(x, d)
        y = list2num(y, d)
        ind = list2index(config[1:])
        image[x, y] = eval('state[' + ind + ']')
        config[-1] += 1
        for n in range(num, 0, -1):
            if config[n] == d:
                config[n] = 0
                config[n-1] += 1
            else:
                break
    if is_rescale:
        image = image / max(abs(image.reshape(-1, ))) * 255
    return image

def state2image_3(state, d, index_x, index_y, is_rescale=False):
    '''
    对于没有用RDM的情况，随机排列，即任意选出一半的qubits决定x轴，另一半的qubits决定y轴
    需要在主程序中添加下列随机排列的代码
    index_list = list(range(num))
    random.shuffle(index_list)
    index_x = index_list[:num_h]
    index_y = index_list[num_h:]
    index_x.sort()
    index_y.sort()
    '''
    num = int(round(np.log(state.size) / np.log(d)))
    if state.ndim != num:
        state = state.reshape(np.ones((num, ), dtype=int) * d)
    num_h = int(round(num/2))
    image = np.zeros((d**num_h, d**num_h))
    config = [0] * (num + 1)
    while config[0] == 0:
        x = list()
        x.append(config[0])   # 这里的第一个维度是用来判断循环是否结束的，因此要先加到x列表里
        for nx in index_x:
            x.append(config[nx+1])
            # nx+1是因为config一共n+1个数，config[0]不是构型，真正的构型是config[1:]
            # 但nx和ny是从0开始就算构型了，因此要+1以便和config对应
        y = list()
        for ny in index_y:
            y.append(config[ny+1])
        x = list2num(x, d)
        y = list2num(y, d)
        ind = list2index(config[1:])
        image[x, y] = eval('state[' + ind + ']')
        config[-1] += 1
        for n in range(num, 0, -1):
            if config[n] == d:
                config[n] = 0
                config[n-1] += 1
            else:
                break
    if is_rescale:
        image = image / max(abs(image.reshape(-1, ))) * 255
    return image


def list2num(x, d):
    num = 0
    length = x.__len__()
    for n in range(0, length):
        num += x[n] * d**(length - n - 1)
    return num


def list2index(x):
    ind = ''
    for n in range(0, x.__len__()):
        ind = ind + str(x[n]) + ','
    return ind[:-1]


def image2rgb(image, if_rescale_1=False):
    # Positive: red; negative: blue
    shape = image.shape
    im = Image.new("RGB", shape)
    for nx in range(0, shape[0]):
        for ny in range(0, shape[1]):
            if if_rescale_1:
                if image[nx, ny] > 0:
                    im.putpixel((nx, ny), (0, 255, 255))
                else:
                    im.putpixel((nx, ny), (255, 0, 255))
            else:
                if image[nx, ny] > 0:
                    im.putpixel((nx, ny), (255 - int(image[nx, ny]), 255, 255))
                else:
                    im.putpixel((nx, ny), (255, 255 + int(image[nx, ny]), 255))
    return im





