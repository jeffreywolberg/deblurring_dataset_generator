import numpy as np
import cv2

def get_psnr(im1, im2):
    # images are of shape (H, W, C)
    assert im1.shape == im2.shape, f"im1.shape: {im1.shape} != im2.shape: {im2.shape}"
    assert len(im1.shape) == 3, f"{im1.shape} must be (H, W, C)"
    assert im1.shape[-1] in [1, 3], f"{im1.shape} must be (H, W, C)"
    
    # dtype checks
    assert im1.dtype == im2.dtype, f"im1.dtype: {im1.dtype} != im2.dtype: {im2.dtype}"
    assert im1.dtype == np.uint8

    mse = np.average((im1.astype(np.float32) - im2.astype(np.float32)) ** 2, axis=None)
    mse += 1e-6 # to avoid div by 0 errors
    psnr = 10 * np.log10(255**2 / mse)

    return psnr

def get_ssim(im1, im2, ksize=11):
    # images are of shape (H, W, C)
    assert im1.shape == im2.shape, f"im1.shape: {im1.shape} != im2.shape: {im2.shape}"
    assert len(im1.shape) == 3, f"{im1.shape} must be (H, W, C)"
    assert im1.shape[-1] in [1, 3], f"{im1.shape} must be (H, W, C)"
    
    # dtype checks
    assert im1.dtype == im2.dtype, f"im1.dtype: {im1.dtype} != im2.dtype: {im2.dtype}"
    assert im1.dtype == np.uint8

    ksize_sq = ksize * ksize
    h, w, ch = im1.shape
    hd = h % ksize # height diff
    wd = w % ksize # width diff 

    # ensure image size is a multiple of ksize
    im1 = im1[hd // 2 : h - ((hd + 1) // 2), wd // 2 : w - ((wd + 1) // 2)]
    im2 = im2[hd // 2 : h - ((hd + 1) // 2), wd // 2 : w - ((wd + 1) // 2)]

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    assert im1.shape[0] % ksize == 0, f"im1.shape: {im1.shape}, ksize: {ksize}"
    assert im1.shape[1] % ksize == 0, f"im1.shape: {im1.shape}, ksize: {ksize}"
    assert im2.shape[0] % ksize == 0, f"im2.shape: {im2.shape}, ksize: {ksize}"
    assert im2.shape[1] % ksize == 0, f"im2.shape: {im2.shape}, ksize: {ksize}"

    h, w, ch = im1.shape

    mu1 = np.zeros((h//ksize, w//ksize, ch))
    mu2 = np.zeros((h//ksize, w//ksize, ch))
    sig1 = np.zeros((h//ksize, w//ksize, ch))
    sig2 = np.zeros((h//ksize, w//ksize, ch))
    sig12 = np.zeros((h//ksize, w//ksize, ch))

    L = 255
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    for i, r in enumerate(range(0, h, ksize)):
        for j, c in enumerate(range(0, w, ksize)):
            mu1[i, j] = np.sum(im1[r:r+ksize, c:c+ksize], axis=(0, 1)) / ksize_sq
            mu2[i, j] = np.sum(im2[r:r+ksize, c:c+ksize], axis=(0, 1)) / ksize_sq
            sig1[i, j] = np.sum((im1[r:r+ksize, c:c+ksize] - mu1[i, j]) ** 2, axis=(0, 1)) / (ksize_sq - 1)
            sig2[i, j] = np.sum((im2[r:r+ksize, c:c+ksize] - mu2[i, j]) ** 2, axis=(0, 1)) / (ksize_sq - 1)
            sig12[i, j] = np.sum((im1[r:r+ksize, c:c+ksize] - mu1[i, j]) * (im2[r:r+ksize, c:c+ksize] - mu2[i, j]) , axis=(0, 1)) / (ksize_sq - 1)

    num = (2 * mu1 * mu2 + C1) * (2 * sig12 + C2)
    denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2)
    ssim_channels = np.average(num / denom, axis=(0, 1)) # average across (ksize x ksize) patches
    ssim = np.average(ssim_channels) # average across channels

    return ssim

if __name__ == "__main__":
    # TODO, fill with your images
    im_path1 = "/Users/jeffreywolberg/Coding/deblurring_project/data/level3/saved_images/GOPR0374_11_00_st_frame-802_n_avg-10.png" 
    im_path2 = "/Users/jeffreywolberg/Coding/deblurring_project/data/level3/saved_images/GOPR0374_11_00_st_frame-802_n_avg-10_gt.png" 
    # im_path2 = "/Users/jeffreywolberg/Coding/deblurring_project/data/level3/saved_images/GOPR0374_11_00_st_frame-802_n_avg-10.png" 

    im1 = cv2.imread(im_path1)
    im2 = cv2.imread(im_path2)

    psnr = get_psnr(im1, im2)
    print(f"psnr: {psnr}")

    ssim = get_ssim(im1, im2, ksize=11)
    print(f"ssim: {ssim}")

