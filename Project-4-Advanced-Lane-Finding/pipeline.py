import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import calibration



path ='/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #4 (Advanced Lane Finding)/CarND-Advanced-Lane-Lines/test_images'
images = glob.glob(path + '/test*.jpg')

#use mtx and dist values calculated for camera calibration:
mtx = calibration.mtx
dist = calibration.dist



def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    explores the direction, or orientation, of the gradient
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir>=thresh[0])&(grad_dir<=thresh[1])]=1
    return dir_binary    



def thres_img(img):
  '''
  combines binary threshold for color transforms and gradient direction
  '''

  #luv: (better for white lanes detection, using l channel)
  luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
  l_channel = luv[:,:,0]
  l_thresh_min = 220
  l_thresh_max = 255
  l_binary = np.zeros_like(l_channel)
  l_binary[(l_channel>=l_thresh_min) & (l_channel<=l_thresh_max)]=1
  # plt.imshow(l_binary, cmap = 'gray')
  # plt.show()

  #lab: (better for yellow lanes detection, using b channel)
  lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  lab_channel = luv[:,:,2]
  lab_thresh_min = 180
  lab_thresh_max = 255
  lab_binary = np.zeros_like(lab_channel)
  lab_binary[(lab_channel>=lab_thresh_min) & (lab_channel<=lab_thresh_max)]=1
  # plt.imshow(lab_binary, cmap = 'gray')
  # plt.show()

  #apply direction of gradient
  dir_binary = dir_threshold(img, sobel_kernel=3, thresh=(0.1, 1.3))

  #combine the binary thresholds
  combined_binary = np.zeros_like(l_binary)
  combined_binary[((lab_binary == 1) & (dir_binary ==1)) | (l_binary == 1) ] = 1
  # plt.imshow(combined_binary, cmap='gray')
  # plt.show()
  return combined_binary




def warp(img):
  '''
  selects areas and applies Perspective Transform to get "bird-eye-view"
  '''
  img_size = (img.shape[1], img.shape[0])

  src = np.float32([[240,720],
                    [575,460],
                    [715,460],
                    [1150,720]])

  dst = np.float32([[240,720],
                    [240,0],
                    [1150,0],
                    [1150,720]])

  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
  return warped


def lines_pixels(img):
  '''
  Finds right and left lane lines using "histogram" peaks search. It divides the image in several sections.
  For each section it detects the peak and for the subsequest sections only the peak in close proximity with the
  previous one is kept. This avoids to consider peaks that are actually not part of the line.
  '''
  right_lane_x = []
  right_lane_y = []
  left_lane_x = []
  left_lane_y = []

  #for left lane:
  hist_left_half = np.sum(img[img.shape[0]/2:,:img.shape[1]/2], axis=0)
  init_peak_left = int(np.argmax(hist_left_half))

  x_val = init_peak_left
  previous_left = x_val
  for i in (range(100)):
    #split the image in several horizontal sections and detect x_val as argmax of histogram peaks
    img_y1 = img.shape[0]-img.shape[0]*i/100
    img_y2 = img.shape[0]-img.shape[0]*(i+1)/100

    histogram = np.sum(img[img_y2:img_y1,:img.shape[1]/2], axis=0)
    x_val = int(np.argmax(histogram))
    y_val = int(720-i*img.shape[0]/100)
    if (y_val == 0 or x_val == 0):
      pass
    #it keeps track of previous "section" peak detection and for the new "section" considers peaks if they are
    #in close proximity with the previous one
    elif (abs(x_val-previous_left) > 100 and not(i == 99) and not(previous_left == 0)):
      pass
    else:
      left_lane_x.append(x_val)
      left_lane_y.append(y_val)
      previous_left = x_val

  #for right lane:
  hist_right_half = np.sum(img[img.shape[0]/2:,img.shape[1]/2:], axis=0)
  init_peak_right = int(np.argmax(hist_right_half)) + 640

  x_val = init_peak_right
  previous_right = x_val
  for i in (range(100)):
    #split the image in several horizontal sections and detect x_val as argmax of histogram peaks
    img_y1 = img.shape[0]-img.shape[0]*i/100
    img_y2 = img.shape[0]-img.shape[0]*(i+1)/100

    histogram = np.sum(img[img_y2:img_y1,(img.shape[1]/2):], axis=0)
    x_val = int(np.argmax(histogram)) + 640
    y_val = int(720-i*img.shape[0]/100)
    if (y_val == 0 or x_val == 640):
      pass
    #it keeps track of previous "section" peak detection and for the new "section" considers peaks if they are
    #in close proximity with the previous one
    elif (abs(x_val-previous_right) > 100 and not(i == 99) and not(previous_right == 0)):
      pass
    else:
      right_lane_x.append(x_val)
      right_lane_y.append(y_val)
      previous_right = x_val
          

  #fit polynomial on the left:
  left_lane_y = np.array(left_lane_y)
  left_lane_x = np.array(left_lane_x)
  left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
  left_fitx = left_fit[0]*left_lane_y**2 + left_fit[1]*left_lane_y + left_fit[2]


  #fit polynomial on the right:
  right_lane_y = np.array(right_lane_y)
  right_lane_x = np.array(right_lane_x)

  right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
  right_fitx = right_fit[0]*right_lane_y**2 + right_fit[1]*right_lane_y + right_fit[2]

  #find position of the car at the end of the y lane (closed to bottom of image)
  right_fitx_position = right_fit[0]*715**2 + right_fit[1]*715 + right_fit[2]
  left_fitx_position = left_fit[0]*715**2 + left_fit[1]*715 + left_fit[2]
  middle_lane = left_fitx_position + (right_fitx_position - left_fitx_position)/2
  
  car_position = (640-middle_lane)*(3.7/700)

  return left_fit, right_fit, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_fitx, right_fitx, car_position

def radius(left_lane_x, left_lane_y, right_lane_x, right_lane_y):
  '''
  calculates curvature of left and right lane
  '''
  #transform from pixel value to meter
  ym_per_pix = 30.0/720.0
  xm_per_pix = 3.7/700.0

  y_eval_left = np.max(left_lane_y)
  y_eval_right = np.max(right_lane_y)
  left_fit_cr = np.polyfit(left_lane_y*ym_per_pix, left_lane_x*xm_per_pix, 2)
  right_fit_cr = np.polyfit(right_lane_y*ym_per_pix, right_lane_x*xm_per_pix, 2)

  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])

  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])

  return left_curverad, right_curverad



def draw_on_img(warped_img, img, image, left_fit, right_fit, left_lane_y, right_lane_y):
  '''
  Draws the lines and an area connecting left and right lines back down onto the original image
  '''
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped_img).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  left_lane_y = np.linspace(np.min(left_lane_y), image.shape[0],10)
  right_lane_y = left_lane_y
  left_fitx = left_fit[0]*left_lane_y**2 + left_fit[1]*left_lane_y + left_fit[2]
  right_fitx = right_fit[0]*left_lane_y**2 + right_fit[1]*left_lane_y + right_fit[2]
  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, left_lane_y]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_lane_y])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
  src = np.float32([[240,720],
                    [575,460],
                    [715,460],
                    [1150,720]])

  dst = np.float32([[240,720],
                    [240,0],
                    [1150,0],
                    [1150,720]])
  Minv = cv2.getPerspectiveTransform(dst, src)
  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
  # Combine the result with the original image
  result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

  return result


###########################
#For SINGLE IMAGE:

def image_pipeline(image):
  '''
  pipeline for single image. It applies distortion correction, binary threshold, perspective transform,
  lines detection and finally draws the detected area down on the original image
  '''
  image = mpimg.imread(image)
  #apply distortion correction to the raw image
  img = cv2.undistort(image, mtx, dist, None, mtx)
  #apply threshold
  thresholded = thres_img(img)
  #apply perspective transform for "bird-eye-view"
  warped_img = warp(thresholded)

  #lines detection
  left_fit, right_fit, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_fitx, right_fitx, car_position= lines_pixels(warped_img)
  left_curverad, right_curverad = radius(left_fitx, left_lane_y, right_fitx, right_lane_y)
  left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
  right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
  #draws area back on the original image 
  result = draw_on_img(warped_img, img, image, left_fit, right_fit, left_lane_y, right_lane_y)

  cv2.putText(result, "Lane curvature: %.2f m" %left_curverad, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  if car_position<0:
    cv2.putText(result, "Car is left of center by: %.2f m" %abs(car_position), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  else:
    cv2.putText(result, "Car is right of center by: %.2f m" %abs(car_position), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  plt.imshow(result)
  # plt.show() 
#############################

#For VIDEO STREAM:

def region_of_interest(img, vertices):
  """
  Applies an image mask.
  Only keeps the region of the image defined by the polygon
  formed from `vertices`. The rest of the image is set to black.
  """
  #defining a blank mask to start with
  mask = np.zeros_like(img)   
  
  if len(img.shape) > 2:
      channel_count = img.shape[2]  
      ignore_mask_color = (255,) * channel_count
  else:
      ignore_mask_color = 255
      
  #filling pixels inside the polygon defined by "vertices" with the fill color    
  cv2.fillPoly(mask, vertices, ignore_mask_color)
  
  #returning the image only where mask pixels are nonzero
  masked_image = cv2.bitwise_and(img, mask)
  
  return masked_image

def radius_curv(xpoints, ypoints):
  '''
  Calculates the curvature at the max y point 
  '''
  ym_per_pix = 30.0/720.0
  xm_per_pix = 3.7/700.0

  y_eval = np.max(ypoints)

  fit_cr = np.polyfit(ypoints*ym_per_pix, xpoints*xm_per_pix, 2)

  curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                             /np.absolute(2*fit_cr[0])
  return curverad


#create a class to keep track of changes from frame to frame
class Line():
    def __init__(self, xpoints, ypoints, fitx, fit):
      self.detected = False 
      self.radius_of_curvature = radius_curv(xpoints, ypoints)
      self.allx= xpoints 
      self.ally = ypoints 
      self.fitx = fitx
      self.fit = fit

frame_count = 0
left_curverad_prior = 0
right_curverad_prior = 0
matching_count_left = 0
matching_count_right = 0

def pipeline(image):
  '''
  pipeline for video. It applies distortion correction, binary threshold, perspective transform,
  lines detection and finally draws the detected area down on the original image frames.
  '''
  global frame_count, left_fit, right_fit, left_curverad_prior, right_curverad_prior, vertices_left, vertices_right, left, right, left_lane_prior, right_lane_prior, matching_count_left, matching_count_right

  frame_count +=1
  
  #apply distortion correction
  img = cv2.undistort(image, mtx, dist, None, mtx)
  #apply binary threshold
  thresholded = thres_img(img)  
  #apply perspective transform for "bird-eye-view"
  warped_img = warp(thresholded)

  #for the first video frame it detects the lines with the "histogram" approach
  if frame_count ==1:
    left_fit, right_fit, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_fitx, right_fitx, car_position= lines_pixels(warped_img)
    left_curverad, right_curverad = radius(left_lane_x, left_lane_y, right_lane_x, right_lane_y)
    
    #set class instances to keep track of values
    left = Line(left_lane_x, left_lane_y, left_fitx, left_fit)
    right = Line(right_lane_x, right_lane_y, right_fitx, right_fit)

    #these values are set for the class instances
    y_pts_left = left.ally
    y_pts_right = right.ally
    x_pts_left = left.allx
    x_pts_right = right.allx
    left_fitx = left.fitx
    right_fitx = right.fitx

    #set the values of curvature of the first frame as "prior" values for the next frames
    left_curverad_prior = left_curverad
    right_curverad_prior = right_curverad

    #set the class instances of left and right lanes as "prior" values for the next frames
    right_lane_prior = right
    left_lane_prior = left 

    #calculates values for 2nd degree polynomial
    left_fit = np.polyfit(y_pts_left, x_pts_left, 2)
    right_fit = np.polyfit(y_pts_right, x_pts_right, 2)

  #for the following 10 frames it detects the lines with the "histogram" approach and verifies "high confidence" for lanes detection
  elif frame_count>1 and frame_count <12:
    try:
      left_fit, right_fit, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_fitx, right_fitx, car_position= lines_pixels(warped_img)
      left_curverad, right_curverad = radius(left_lane_x, left_lane_y, right_lane_x, right_lane_y)

      #the values of the left lane curvature and heading are evaluated versus the prior ones in the previous frame. If the values are similar (within 10%) we 
      #consider it a "lane detection"
      if (abs(left_fit[0] - left_lane_prior.fit[0]) < 0.3) & (abs(left_fit[1] - left_lane_prior.fit[1]) < 0.3) :
        matching_count_left += 1
        left = Line(left_lane_x, left_lane_y, left_fitx, left_fit)
        left.detected = True
      else:
        left = left_lane_prior

      #the values of the right lane curvature and heading are evaluated versus the prior ones in the previous frame. If the values are similar (within 10%) we 
      #consider it a "lane detection"
      if (abs(right_fit[0] - right_lane_prior.fit[0]) < 0.3) & (abs(right_fit[1] - right_lane_prior.fit[1]) < 0.3):
        matching_count_right += 1
        right = Line(right_lane_x, right_lane_y, right_fitx, right_fit)
        right.detected = True
      else:
        right = right_lane_prior

      #these values are set for the class instances
      y_pts_left = left.ally
      y_pts_right = right.ally
      x_pts_left = left.allx
      x_pts_right = right.allx
      left_fitx = left.fitx
      right_fitx = right.fitx

      #set values of the current frame as "prior" values for the next frame
      left_curverad_prior = left_curverad
      right_curverad_prior = right_curverad
      left_lane_prior = left
      right_lane_prior = right

      #calculates values for 2nd degree polynomial
      left_fit = np.polyfit(y_pts_left, x_pts_left, 2)
      right_fit = np.polyfit(y_pts_right, x_pts_right, 2)

    except TypeError:
      print('error in the frame')
      left = left_lane_prior
      right = right_lane_prior 
      left_curverad = left_curverad_prior
      right_curverad = right_curverad_prior
      #
      y_pts_left = left.ally
      y_pts_right = right.ally
      x_pts_left = left.allx
      x_pts_right = right.allx
      left_fitx = left.fitx
      right_fitx = right.fitx
      left_fit = left.fit
      right_fit = right.fit


      right_fitx_position = right_fit[0]*715**2 + right_fit[1]*715 + right_fit[2]
      left_fitx_position = left_fit[0]*715**2 + left_fit[1]*715 + left_fit[2]
      middle_lane = left_fitx_position + (right_fitx_position - left_fitx_position)/2
      
      car_position = (640-middle_lane)*(3.7/700)

  #for the rest of the frames after the first ones we use the position of the detected lane in the prior frame as
  #a starting point for pixels detection. If there is "high confidence" of lanes detection in the prior ten frames 
  #then a "window" around the prior lanes positions is drafted and only pixels within that window are 
  #considered as part of the lanes. (so, no histogram search, but "nonzero" approach for pixels detection)
  else:
    #matching_count is how many times in the prior ten frames the lanes where detected. It's considered as a value of "high confidence"
    #for subsequent frames. We also evaluate that in the "prior" frame the "nonzero" values were not empty.
    if matching_count_left==10 & matching_count_right==10 and left_lane_prior.fitx != [] and right_lane_prior.fitx !=[]:
      maximum_x_left = np.max(left_lane_prior.allx)
      minimum_x_left = np.min(left_lane_prior.allx)

      maximum_x_right = np.max(right_lane_prior.allx)
      minimum_x_right = np.min(right_lane_prior.allx)

      vertices_left = np.array([[ (minimum_x_left-50, 720), (minimum_x_left-50, 0), (maximum_x_left+50, 0), (maximum_x_left+50, 720)]])
      vertices_right = np.array([[ (minimum_x_right-50, 720), (minimum_x_right-50, 0), (maximum_x_right+50, 0), (maximum_x_right+50, 720)]])
      
      #masking an area (a rectangle) around the prior lanes detected)
      masked_rectangle_left = region_of_interest(warped_img, vertices_left)
      y_pts_left, x_pts_left = np.nonzero(masked_rectangle_left)
      y_pts_left = np.array(y_pts_left)
      x_pts_left = np.array(x_pts_left)

      masked_rectangle_right = region_of_interest(warped_img, vertices_right)
      y_pts_right, x_pts_right = np.nonzero(masked_rectangle_right)
      y_pts_right = np.array(y_pts_right)
      x_pts_right = np.array(x_pts_right)

      #fit a polynomial for the pixels detected
      left_fit = np.polyfit(y_pts_left, x_pts_left, 2)
      left_fitx = left_fit[0]*y_pts_left**2 + left_fit[1]*y_pts_left + left_fit[2]

      right_fit = np.polyfit(y_pts_right, x_pts_right, 2)
      right_fitx = right_fit[0]*y_pts_right**2 + right_fit[1]*y_pts_right + right_fit[2]

      #set the current lanes as "prior" for the following frames
      left_lane_prior = Line(x_pts_left, y_pts_left,left_fitx, left_fit)
      right_lane_prior = Line(x_pts_right, y_pts_right, right_fitx, right_fit)

      #calculates curvature and car position values
      left_curverad, right_curverad = radius(x_pts_left, y_pts_left, x_pts_right, y_pts_right)

      right_fitx_position = right_fit[0]*715**2 + right_fit[1]*715 + right_fit[2]
      left_fitx_position = left_fit[0]*715**2 + left_fit[1]*715 + left_fit[2]
      middle_lane = left_fitx_position + (right_fitx_position - left_fitx_position)/2
      
      car_position = (640-middle_lane)*(3.7/700)

    #if no "high confidence" in the first frames is achieved, then we use the "histogram" search approach for pixels detection
    else:
      try:
        left_fit, right_fit, left_lane_x, left_lane_y, right_lane_x, right_lane_y, left_fitx, right_fitx, car_position= lines_pixels(warped_img)
        left_curverad, right_curverad = radius(left_lane_x, left_lane_y, right_lane_x, right_lane_y)

        y_pts_left = left_lane_y
        y_pts_right = right_lane_y
        x_pts_left = left_lane_x
        x_pts_right = right_lane_x
   
        left_fit = np.polyfit(y_pts_left, x_pts_left, 2)
        right_fit = np.polyfit(y_pts_right, x_pts_right, 2)

      except TypeError:
        print('error in the frame')
        left = left_lane_prior
        right = right_lane_prior 
        left_curverad = left_curverad_prior
        right_curverad = right_curverad_prior
        #
        y_pts_left = left.ally
        y_pts_right = right.ally
        x_pts_left = left.allx
        x_pts_right = right.allx
        left_fitx = left.fitx
        right_fitx = right.fitx
        left_fit = left.fit
        right_fit = right.fit


        right_fitx_position = right_fit[0]*715**2 + right_fit[1]*715 + right_fit[2]
        left_fitx_position = left_fit[0]*715**2 + left_fit[1]*715 + left_fit[2]
        middle_lane = left_fitx_position + (right_fitx_position - left_fitx_position)/2
        
        car_position = (640-middle_lane)*(3.7/700)

  result = draw_on_img(warped_img, img, image, left_fit, right_fit, y_pts_left, y_pts_right)

  # draws values for curvature and car position for each video frame
  cv2.putText(result, "Lane curvature: %.2f m" %left_curverad, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  if car_position<0:
    cv2.putText(result, "Car is left of center by: %.2f m" %abs(car_position), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  else:
    cv2.putText(result, "Car is right of center by: %.2f m" %abs(car_position), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
  
  return result 
  


from moviepy.editor import VideoFileClip
from IPython.display import HTML
vid_output = 'project_video.mp4'
clip1 = VideoFileClip('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #4 (Advanced Lane Finding)/CarND-Advanced-Lane-Lines/project_video.mp4')
vid_clip = clip1.fl_image(pipeline) 
vid_clip.write_videofile(vid_output, audio=False)

