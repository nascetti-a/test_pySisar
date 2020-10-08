import numpy as np


def find_corresponding_point(model_a, model_b, x, y, z):
    """
    Finds corresponding points in the second image, given the heights.
    Arguments:
        model_a, model_b: two instances of the rpcm.RPCModel class, or of
            the projective_model.ProjModel class
        x, y, z: three 1D numpy arrrays, of the same length. x, y are the
        coordinates of pixels in the image, and z contains the altitudes of the
        corresponding 3D point.
    Returns:
        xp, yp, z: three 1D numpy arrrays, of the same length as the input. xp,
            yp contains the coordinates of the projection of the 3D point in image
            b.
    """
    t1, t2 = model_a.localization(x, y, z)
    xp, yp = model_b.projection(t1, t2, z)
    return (xp, yp, z)


def compute_height(model_a, model_b, x1, y1, x2, y2):
    """
    Computes the height of a point given its location inside two images.
    Arguments:
        model_a, model_b: two instances of the rpcm.RPCModel class, or of
            the projective_model.ProjModel class
        x1, y1: two 1D numpy arrrays, of the same length, containing the
            coordinates of points in the first image.
        x2, y2: two 2D numpy arrrays, of the same length, containing the
            coordinates of points in the second image.
    Returns:
        a 1D numpy array containing the list of computed heights.
    """
    n = len(x1)
    h0 = np.zeros(n)
    h0_inc = h0
    p2 = np.vstack([x2, y2]).T
    HSTEP = 1
    err = np.zeros(n)

    for i in range(100):
        print(h0)
        tx, ty, tz = find_corresponding_point(model_a, model_b, x1, y1, h0)
        r0 = np.vstack([tx,ty]).T
        tx, ty, tz = find_corresponding_point(model_a, model_b, x1, y1, h0+HSTEP)
        r1 = np.vstack([tx,ty]).T
        a = r1 - r0
        b = p2 - r0
        # implements: h0_inc = dot(a,b) / dot(a,a)
        # For some reason, the formulation below causes massive memory leaks on
        # some systems.
        # h0_inc = np.divide(np.diag(np.dot(a, b.T)), np.diag(np.dot(a, a.T)))
        # Replacing with the equivalent:
        diagabdot = np.multiply(a[:, 0], b[:, 0]) + np.multiply(a[:, 1], b[:, 1])
        diagaadot = np.multiply(a[:, 0], a[:, 0]) + np.multiply(a[:, 1], a[:, 1])
        h0_inc = np.divide(diagabdot, diagaadot)
#        if np.any(np.isnan(h0_inc)):
#            print(x1, y1, x2, y2)
#            print(a)
#            return h0, h0*0
        # implements:   q = r0 + h0_inc * a
        q = r0 + np.dot(np.diag(h0_inc), a)
        # implements: err = sqrt(dot(q-p2, q-p2))
        tmp = q-p2
        err =  np.sqrt(np.multiply(tmp[:, 0], tmp[:, 0]) + np.multiply(tmp[:, 1], tmp[:, 1]))
#       print(np.arctan2(tmp[:, 1], tmp[:, 0])) # for debug
#       print(err) # for debug
        h0 = np.add(h0, h0_inc*HSTEP)

        # implements: if fabs(h0_inc) < 0.0001:
        if np.max(np.fabs(h0_inc)) < 0.001:
            break

    return h0, err


# def align_and_compute_disp(img_pre, img_post, rotate=False, img_enhance=False):
#
#     if img_enhance:
#         img_pre = enhance(img_pre)
#         img_post = enhance(img_post)
#
#     im1_gray = img_pre #cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
#     im2_gray = img_post #cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)
#
#     # Find size of image1
#     sz = img_pre.shape
#
#     # Define the motion model
#     warp_mode = cv2.MOTION_TRANSLATION  #cv2.MOTION_EUCLIDEAN  # cv2.MOTION_TRANSLATION or cv2.MOTION_EUCLIDEAN
#
#
#
#     # Define 2x3 or 3x3 matrices and initialize the matrix to identity
#     if warp_mode == cv2.MOTION_HOMOGRAPHY:
#         warp_matrix = np.eye(3, 3, dtype=np.float32)
#     else:
#         warp_matrix = np.eye(2, 3, dtype=np.float32)
#
#     # Specify the number of iterations.
#     number_of_iterations = 500;
#
#     # Specify the threshold of the increment
#     # in the correlation coefficient between two iterations
#     termination_eps = 1e-10;
#
#     # Define termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
#
#     # Run the ECC algorithm. The results are stored in warp_matrix.
#
#     print(type(im1_gray))
#     (cc, warp_matrix) = cv2.findTransformECC(im1_gray.astype("float32"), im2_gray.astype("float32"), warp_matrix, warp_mode, criteria, None, 1)
#
#     print("Epipolar Transformation matrix: ", warp_matrix)
#
#     print(cc)
#
#     # print(warp_matrix)
#
#     if warp_mode == cv2.MOTION_HOMOGRAPHY:
#         # Use warpPerspective for Homography
#         img_post_aligned = cv2.warpPerspective(img_post, warp_matrix, (sz[1], sz[0]),
#                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     else:
#         # Use warpAffine for Translation, Euclidean and Affine
#         img_post_aligned = cv2.warpAffine(img_post, warp_matrix, (sz[1], sz[0]),
#                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
#
#     filt_disp, conf = compute_disp(img_pre, img_post_aligned, rotate)
#
#     # disp, disp2, filt_disp = compute_disp(im1_gray, im2_gray, rotate)
#
#     # flow_disp, rgb_flow = compute_optical_flow(img_pre, img_post_aligned)
#
#     flow, rgb_flow = optical_flow(img_pre, img_post_aligned)
#
#     #print(disp.shape)
#
#     #fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 10))
#     # fig2.suptitle(f'{disaster_name} - {idx}')
#
#     #ax3.imshow(flow[..., 1])
#     #ax3.set_axis_off()
#     # overlay_label_patches(ax3, data_pre['features'])
#
#
#     #ax4.set_axis_off()
#
#     #fig2.tight_layout()
#     #fig2.subplots_adjust(top=0.88)
#     plt.subplot(131), plt.imshow(flow[..., 0])
#     plt.subplot(132), plt.imshow(flow[..., 1])
#     plt.subplot(133), plt.imshow(filt_disp, vmin=-128, vmax=128)
#     plt.show()
#
#     return flow[..., 0], flow[..., 1], filt_disp, conf, img_post_aligned
#
#
# def compute_disp(imgL, imgR, rotate=False):
#
#     imgL = imgL.astype("uint8")
#     imgR = imgR.astype("uint8")
#
#     if rotate:
#         imgL = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
#         imgR = cv2.rotate(imgR, cv2.ROTATE_90_CLOCKWISE)
#
#     # SGBM Parameters -----------------
#     window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#
#     left_matcher = cv2.StereoSGBM_create(
#         minDisparity=-64,
#         numDisparities=128,  # max_disp has to be dividable by 16 f. E. HH 192, 256
#         blockSize=7,
#         P1=8 * 3 * window_size ** 2,
#         # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#         P2=32 * 3 * window_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=0,
#         speckleRange=2,
#         preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#
#     right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
#
#     # FILTER Parameters
#     lmbda = 80000
#     sigma = 1.2
#     visual_multiplier = 1.0
#
#     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
#     wls_filter.setLambda(lmbda)
#     wls_filter.setSigmaColor(sigma)
#
#     print('computing disparity...')
#     displ = left_matcher.compute(imgL, imgR).astype(np.float32) #/16
#     dispr = right_matcher.compute(imgR, imgL).astype(np.float32) #/16
#     displ = np.int16(displ)
#     dispr = np.int16(dispr)
#     filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
#
#     # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#     # filteredImg = np.uint8(filteredImg)
#
#     if rotate:
#         filteredImg = cv2.rotate(filteredImg, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     return filteredImg, wls_filter.getConfidenceMap()
#
#
# def optical_flow(one, two):
#     """
#     method taken from (https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4)
#     """
#     one_g = one #cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
#     two_g = two #cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
#     hsv = np.zeros((one_g.shape[0], one_g.shape[1], 3))
#     # set saturation
#     #hsv[:, :, 1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:, :, 1]
#     # obtain dense optical flow paramters
#     flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
#                                         pyr_scale=0.5, levels=3, winsize=15,
#                                         iterations=2,
#                                         poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
#     # convert from cartesian to polar
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     # hue corresponds to direction
#     hsv[:, :, 0] = ang * (180 / np.pi / 2)
#     # value corresponds to magnitude
#     hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     # convert HSV to int32's
#     hsv = np.asarray(hsv, dtype=np.float32)
#     rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#     return flow, rgb_flow
#
#
