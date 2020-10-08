import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pySISAR.config import cfg
#from pySISAR import save_raster_as_geotiff
from pySISAR.ortho_tool import save_raster_as_geotiff


def align_and_compute_disp(img_pre, img_post, rotate=False):  # img_enhance=False):

    # if img_enhance:
    #    img_pre = enhance(img_pre)
    #    img_post = enhance(img_post)

    im1_gray = img_pre  # cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    im2_gray = img_post  # cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = img_pre.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION  # cv2.MOTION_EUCLIDEAN  # cv2.MOTION_TRANSLATION or cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 500;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    print(type(im1_gray))
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype("float32"), im2_gray.astype("float32"), warp_matrix, warp_mode, criteria, None, 1)

    print("GrEI (Ground Epipolar Image Transformation matrix: ")
    print(warp_matrix)
    print(cc)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        img_post_aligned = cv2.warpPerspective(img_post, warp_matrix, (sz[1], sz[0]),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        img_post_aligned = cv2.warpAffine(img_post, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    if cfg['dense_matching_method'] == 'SGM':

        filt_disp, conf = compute_disp(img_pre, img_post_aligned, rotate)

        plt.imshow(filt_disp, vmin=np.percentile(filt_disp, 2.5), vmax=np.percentile(filt_disp, 97.5))
        plt.show()

        return filt_disp, conf

    elif cfg['dense_matching_method'] == 'FLOW':

        flow, rgb_flow = optical_flow(img_pre, img_post_aligned)

        plt.subplot(121), plt.imshow(flow[..., 0], vmin=np.percentile(flow[..., 0], 2.5), vmax=np.percentile(flow[..., 0], 97.5))
        plt.subplot(122), plt.imshow(flow[..., 1], vmin=np.percentile(flow[..., 1], 2.5), vmax=np.percentile(flow[..., 1], 97.5))
        plt.show()

        return flow[..., 0], flow[..., 1]

    else:
        print("ERROR: Wrong matching method")
        sys.exit(1)


def compute_disp(imgL, imgR, rotate=False):

    imgL = imgL.astype("uint8")
    imgR = imgR.astype("uint8")

    if rotate:
        imgL = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
        imgR = cv2.rotate(imgR, cv2.ROTATE_90_CLOCKWISE)

    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-32,
        numDisparities=64,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=7,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR).astype(np.float32) #/16
    dispr = right_matcher.compute(imgR, imgL).astype(np.float32) #/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    # filteredImg = np.uint8(filteredImg)

    if rotate:
        filteredImg = cv2.rotate(filteredImg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    grid = cfg['grid']

    save_raster_as_geotiff(filteredImg, grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat,
                           cfg['temporary_dir'] + "/disp_SGM.tiff")

    save_raster_as_geotiff(wls_filter.getConfidenceMap(), grid.ul_lon, grid.ul_lat, grid.lr_lon, grid.lr_lat,
                           cfg['temporary_dir'] + "/conf_SGM.tiff")

    return filteredImg, wls_filter.getConfidenceMap()


def optical_flow(one, two):

    one_g = one # cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = two # cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((one_g.shape[0], one_g.shape[1], 3))
    # set saturation
    # hsv[:, :, 1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:, :, 1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow, rgb_flow


