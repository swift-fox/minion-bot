import torch, cv2
from jetbot import Robot
from threading import Timer
from ultralytics import YOLO

# Set the object to follow
obj_follow = 'bottle'

# List of objects recognized by the model
obj_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

follow_idx = obj_names.index(obj_follow)

# Load the model
model = YOLO('yolov8n.pt').to('cuda')

# Camera settings
capture_width = 1280
capture_height = 720
framerate = 6
flip_method = 0
video_width = 320
video_height = 180

# Get the camera
gst_pipeline = (f'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! '
                f'nvvidconv flip-method={flip_method} ! video/x-raw, width=(int){video_width}, height=(int){video_height}, format=(string)BGRx ! '
                f'videoconvert ! video/x-raw, format=(string)BGR ! appsink')
camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Motor settings
move_speed = 0.2
turn_speed = 0.1

robot = Robot()

# Make sure motors don't run indefinitely
stop_timer = Timer(0, robot.stop)
def set_timer(duration):
    global stop_timer

    if stop_timer.is_alive():
        stop_timer.cancel()

    if duration > 0:
        stop_timer = Timer(duration, robot.stop)
        stop_timer.start()

# Motion states
STOP, TURNING, MOVING = range(3)
state = STOP

# State transfer functions
def turn(position):
    global state
    state = TURNING
    set_timer(0.1)
    robot.right(turn_speed if position > 0 else -turn_speed)

def move(distance):
    global state
    state = MOVING
    set_timer(0.5)
    robot.backward(move_speed if distance > 0 else -move_speed)

def stop():
    global state
    state = STOP
    set_timer(0)
    robot.stop()

# State transfer conditions
turn_begin = 0.2
turn_end = 0.1

move_begin_far = 0.09
move_end_far = 0.12
move_end_near = 0.15
move_begin_near = 0.18
move_center = (move_end_near + move_end_far) / 2

def need_turn(position):
    return abs(position) > turn_begin

def turn_done(position):
    return abs(position) < turn_end

def need_move(size):
    return size < move_begin_far or size > move_begin_near

def move_done(size):
    return size > move_end_far and size < move_end_near

# The main loop
while True:
    _, image = camera.read()

    results = model(image, classes=[follow_idx], verbose=False)
    boxes = results[0].boxes

    for i in range(len(boxes.cls)):
        x, y, w, h = boxes.xywh[i]
        score = boxes.conf[i]

        position = x * 2 / video_width - 1
        size = w / video_width

        # Motion control state machine
        if state == STOP:
            if need_turn(position):
                turn(position)
            elif need_move(size):
                move(size - move_center)
        elif state == TURNING:
            if not turn_done(position):
                turn(position)
            elif need_move(size):
                move(size - move_center)
            else:
                stop()
        elif state == MOVING:
            if need_turn(position):
                turn(position)
            elif not move_done(size):
                move(size - move_center)
            else:
                stop()

        break
    else:
        # Stop the motors if no object found
        stop()
