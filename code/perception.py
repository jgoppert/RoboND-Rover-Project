import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_segmenter(img):
    """
    More advanced version of color_thresh
    that does entire image segmentation

    @param: img: the input image
    @return: dict with images for each type of background
        and a composite for visualization
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5,5),np.uint8)

    thresh_rock = cv2.inRange(
        hsv,
        np.array([0, 200, 100], dtype=np.float),
        np.array([100, 255, 255], dtype=np.float))
    thresh_rock = cv2.morphologyEx(thresh_rock, cv2.MORPH_CLOSE, kernel)
    thresh_rock = thresh_rock > 0

    thresh_ground = cv2.inRange(hsv,
                                np.array([0, 0, 160], dtype=np.float),
                                np.array([255, 255, 255], dtype=np.float))
    thresh_ground[thresh_rock] = 0 # rocks are not ground
    thresh_ground = cv2.morphologyEx(thresh_ground, cv2.MORPH_CLOSE, kernel)
    thresh_ground = thresh_ground > 0

    thresh_obstacle = 1 - thresh_ground

    thresh_img = np.zeros_like(img)

    
    ground_x, ground_y = thresh_ground.nonzero()
    thresh_img[ground_x, ground_y,:] = [0, 100, 0]

    obstacle_x, obstacle_y = thresh_obstacle.nonzero()
    thresh_img[obstacle_x, obstacle_y,:] = [100, 0, 0]
    
    rock_x, rock_y = thresh_rock.nonzero()
    thresh_img[rock_x, rock_y,:] = [200, 200, 0]

    return {
        'rock': thresh_rock,
        'ground': thresh_ground,
        'obstacle': thresh_obstacle,
        'img': thresh_img
    }

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = np.deg2rad(yaw)
    xpix_rotated = xpix*np.cos(yaw_rad) - ypix*np.sin(yaw_rad)
    ypix_rotated = xpix*np.sin(yaw_rad) + ypix*np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = xpos + xpix_rot/scale
    ypix_translated = ypos + ypix_rot/scale
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    img = Rover.img
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    roll = Rover.roll
    pitch = Rover.pitch

    # 1) Define source and destination points for perspective transform

    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 

    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6

    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    seg = color_segmenter(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image = seg['img']

    # 5) Convert map image pixel values to rover-centric coords
    navigable_pix_x, navigable_pix_y = rover_coords(seg['ground'])
    obstacle_pix_x, obstacle_pix_y = rover_coords(seg['obstacle'])
    rock_pix_x, rock_pix_y = rover_coords(seg['rock'])

    # 6) Convert rover-centric pixel values to world coordinates
    world_size = 200
    scale = 10
    obstacle_x_world, obstacle_y_world = pix_to_world(
        obstacle_pix_x, obstacle_pix_y,
        xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(
        rock_pix_x, rock_pix_y,
        xpos, ypos, yaw, world_size, scale)

    if len(rock_x_world) > 0:
        goal_dist, goal_angle = to_polar_coords(np.mean(rock_pix_x), np.mean(rock_pix_y))
        Rover.goal_dist = goal_dist
        Rover.goal_angle = goal_angle


    navigable_x_world, navigable_y_world = pix_to_world(
        navigable_pix_x, navigable_pix_y,
        xpos, ypos, yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)

    # only update map if we have small roll/pitch so that the 
    # perspective transform is valid
    if roll > 180:
        roll -= 360
    if pitch > 180:
        pitch -= 360
    if Rover.mode != 'pickup' and np.abs(roll) < 1 and np.abs(pitch) < 1:
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dist, angles = to_polar_coords(navigable_pix_x, navigable_pix_y)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    return Rover
