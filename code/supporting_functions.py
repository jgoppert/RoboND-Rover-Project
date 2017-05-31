import base64
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class RoverState:
    """
    Define RoverState() class to retain rover state parameters
    """
    def __init__(self):
        self.start_time = None  # type: float
        # To record the start time of navigation
        self.total_time = None  # type: float
        # To record total duration of navigation
        self.img = None  # type: np.ndarray
        # Current camera image
        self.pos = None  # type: np.ndarray
        #  Current position (x, y)
        self.yaw = None  # type: float
        # Current yaw angle
        self.pitch = None  # type: float
        # Current pitch angle
        self.roll = None  # type: float
        # Current roll angle
        self.vel = None  # type: float
        # Current velocity
        self.steer = 0  # type: float
        # Current steering angle
        self.throttle = 0  # type: float
        # Current throttle value
        self.brake = 0  # type: float
        # Current brake value
        self.nav_angles = None  # type: np.ndarray
        # Angles of navigable terrain pixels
        self.nav_dists = None  # type: np.ndarray
        # Distances of navigable terrain pixels
        self.ground_truth = None  # type: np.ndarray
        #  Ground truth worldmap
        self.mode = 'forward'  # type: str
        # Current mode (can be forward or stop)
        self.throttle_set = 0.2  # type: float
        # Throttle setting when accelerating
        self.brake_set = 10  # type: float
        # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50  # type: int
        # Threshold to initiate stopping
        self.go_forward = 500  # type: int
        # Threshold to go forward again
        self.max_vel = 2  # type: float
        # Maximum velocity (meters/second)

        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float)  # type: np.ndarray

        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float)  # type: np.ndarray
        self.samples_pos = None  # type: np.ndarray
        # To store the actual sample positions
        self.samples_found = None  # type: np.ndarray
        # To count the number of samples found
        self.near_sample = False  # type: bool
        # Will be set to telemetry value data["near_sample"]
        self.picking_up = False  # type: bool
        #  Will be set to telemetry value data["picking_up"]
        self.send_pickup = False  # type: bool
        # Set to True to trigger rock pickup


def update_rover(rover: RoverState, data: dict):
    # Initialize start time and sample positions
    if rover.start_time is None:
        rover.start_time = time.time()
        rover.total_time = 0
        samples_xpos = np.int_([np.float(pos.strip()) for pos in data["samples_x"].split(',')])
        samples_ypos = np.int_([np.float(pos.strip()) for pos in data["samples_y"].split(',')])
        rover.samples_pos = (samples_xpos, samples_ypos)
        rover.samples_found = np.zeros((len(rover.samples_pos[0]))).astype(np.int)
    # Or just update elapsed time
    else:
        tot_time = time.time() - rover.start_time
        if np.isfinite(tot_time):
            rover.total_time = tot_time
    # Print out the fields in the telemetry data dictionary
    print(data.keys())
    # The current speed of the rover in m/s
    rover.vel = np.float(data["speed"])
    # The current position of the rover
    rover.pos = np.fromstring(data["position"], dtype=float, sep=',')
    # The current yaw angle of the rover
    rover.yaw = np.float(data["yaw"])
    # The current yaw angle of the rover
    rover.pitch = np.float(data["pitch"])
    # The current yaw angle of the rover
    rover.roll = np.float(data["roll"])
    # The current throttle setting
    rover.throttle = np.float(data["throttle"])
    # The current steering angle
    rover.steer = np.float(data["steering_angle"])
    # Near sample flag
    rover.near_sample = np.int(data["near_sample"])
    # Picking up flag
    rover.picking_up = np.int(data["picking_up"])

    print('speed =', rover.vel, 'position =', rover.pos, 'throttle =',
          rover.throttle, 'steer_angle =', rover.steer, 'near_sample', rover.near_sample,
          'picking_up', data["picking_up"])

    # Get the current image from the center camera of the rover
    img_string = data["image"]
    image = Image.open(BytesIO(base64.b64decode(img_string)))
    rover.img = np.asarray(image)

    # Return updated Rover and separate image for optional saving
    return rover, image


# Define a function to create display output given worldmap results
def create_output_images(rover: RoverState):
    # Create a scaled map for plotting and clean up obs/nav pixels a bit
    if np.max(rover.worldmap[:, :, 2]) > 0:
        nav_pix = rover.worldmap[:, :, 2] > 0
        navigable = rover.worldmap[:, :, 2] * (255 / np.mean(rover.worldmap[nav_pix, 2]))
    else:
        navigable = rover.worldmap[:, :, 2]
    if np.max(rover.worldmap[:, :, 0]) > 0:
        obs_pix = rover.worldmap[:, :, 0] > 0
        obstacle = rover.worldmap[:, :, 0] * (255 / np.mean(rover.worldmap[obs_pix, 0]))
    else:
        obstacle = rover.worldmap[:, :, 0]

    likely_nav = navigable >= obstacle
    obstacle[likely_nav] = 0
    plotmap = np.zeros_like(rover.worldmap)
    plotmap[:, :, 0] = obstacle
    plotmap[:, :, 2] = navigable
    plotmap = plotmap.clip(0, 255)
    # Overlay obstacle and navigable terrain map with ground truth map
    map_add = cv2.addWeighted(plotmap, 1, rover.ground_truth, 0.5, 0)

    # Check whether any rock detections are present in worldmap
    rock_world_pos = rover.worldmap[:, :, 1].nonzero()
    # If there are, we'll step through the known sample positions
    # to confirm whether detections are real
    if rock_world_pos[0].any():
        rock_size = 2
        for idx in range(len(rover.samples_pos[0]) - 1):
            test_rock_x = rover.samples_pos[0][idx]
            test_rock_y = rover.samples_pos[1][idx]
            rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1]) ** 2 +
                                        (test_rock_y - rock_world_pos[0]) ** 2)
            # If rocks were detected within 3 meters of known sample positions
            # consider it a success and plot the location of the known
            # sample on the map
            if np.min(rock_sample_dists) < 3:
                rover.samples_found[idx] = 1
                map_add[test_rock_y - rock_size:test_rock_y + rock_size,
                        test_rock_x - rock_size:test_rock_x + rock_size, :] = 255

    # Calculate some statistics on the map results
    # First get the total number of pixels in the navigable terrain map
    tot_nav_pix = np.float(len((plotmap[:, :, 2].nonzero()[0])))
    # Next figure out how many of those correspond to ground truth pixels
    good_nav_pix = np.float(len(((plotmap[:, :, 2] > 0) & (rover.ground_truth[:, :, 1] > 0)).nonzero()[0]))
    # Next find how many do not correspond to ground truth pixels
    # bad_nav_pix = np.float(len(((plotmap[:, :, 2] > 0) & (Rover.ground_truth[:, :, 1] == 0)).nonzero()[0]))
    # Grab the total number of map pixels
    tot_map_pix = np.float(len((rover.ground_truth[:, :, 1].nonzero()[0])))
    # Calculate the percentage of ground truth map that has been successfully found
    perc_mapped = round(100 * good_nav_pix / tot_map_pix, 1)
    # Calculate the number of good map pixel detections divided by total pixels
    # found to be navigable terrain
    if tot_nav_pix > 0:
        fidelity = round(100 * good_nav_pix / tot_nav_pix, 1)
    else:
        fidelity = 0
    # Flip the map for plotting so that the y-axis points upward in the display
    map_add = np.flipud(map_add).astype(np.float32)
    # Add some text about map and rock sample detection results
    cv2.putText(map_add, "Time: " + str(np.round(rover.total_time, 1)) + ' s', (0, 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Mapped: " + str(perc_mapped) + '%', (0, 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Fidelity: " + str(fidelity) + '%', (0, 40),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(map_add, "Rocks Found: " + str(np.sum(rover.samples_found)), (0, 55),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    # Convert map and vision image to base64 strings for sending to server
    pil_img = Image.fromarray(map_add.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

    pil_img = Image.fromarray(rover.vision_image.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded_string1, encoded_string2
