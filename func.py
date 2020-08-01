def change_hist_v2(img, bits=8):
    min = np.min(img)
    max = np.max(img)
    dif = max - min

    step = (2**bits - 1) / dif

    img2 = (img - min) * step
    img2 = np.asarray(img2, dtype=f'uint{bits}')

    return img2
    
    
def simple_segmentation(img):
    img2 = img - np.min(img)
    img2[img2 > 255] = 255

    return img2


def mask_applying(img, mask, bits=8, depth=4):
    img2 = np.zeros((img.shape[0], img.shape[1], 3))
    img2[..., 1] = img
    img2[..., 0] = img

    mask = mask / (2**bits - 1)
    mask = 1 - mask
    mask = ((mask ** depth) * (2**bits - 1))
    img2[..., 2] = mask

    return img2.astype(f'uint{bits}')
    

img = cv2.imread(filename, -1)
mask = simple_segmentation(img)
img = change_hist_v2(img, bits=8)
img = mask_applying(img, mask, bits=8, depth=10)