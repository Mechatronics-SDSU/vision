import vision.vision.Yolov5Detection as yv5
import vision.vision.Socket_Client as Socket_Client
import argparse
import math
import cv2
import vision.vision.gui_helper as gui_helper
import multiprocessing
import time

from multiprocessing import Value

'''
    discord: @kialli
    github: @kchan5071
    
    main class to run vision code
    
    usage:
    initialize class
    run loop
    
'''


import_success = True

try:
    import pyzed.sl as sl 
    from vision.vision.Zed_Wrapper import Zed
except:
    print("Zed library not found")
    import_success = False

parser = argparse.ArgumentParser()
parser.add_argument('-host_ip', help='ip to send images to', required=False)
parser.add_argument('-port', help='port to send images over', required=False)
parser.add_argument('-show_boxes',help='boolean to show object detection boxes', required=False)
parser.add_argument('-model_name', help='model to run on', required=False)
parser.add_argument('-show_depth', help='show depth map', required=False)
parser.add_argument('-get_depth', help='get depth map', required=False)
args = parser.parse_args()


class VideoRunner:

    def __init__(self, linear_acceleration_x , linear_acceleration_y, linear_acceleration_z, 
                angular_velocity_x, angular_velocity_y, angular_velocity_z, 
                orientation_x, orientation_y, orientation_z, 
                depth,
                offset_x, offset_y):
        self.zed = Zed()
        self.cap = None
        self.linear_acceleration_x = linear_acceleration_x
        self.linear_acceleration_y = linear_acceleration_y
        self.linear_acceleration_z = linear_acceleration_z
        self.angular_velocity_x = angular_velocity_x
        self.angular_velocity_y = angular_velocity_y
        self.angular_velocity_z = angular_velocity_z
        self.orientation_x = orientation_x
        self.orientation_y = orientation_y
        self.orientation_z = orientation_z
        self.depth = depth
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        self.detection = None
        self.skip_frames = 0
        self.frame_count = self.skip_frames

    
    #gets median of all objects, then returns the closest ones
    def get_nearest_object(self, results):
        nearest_object = math.inf
        nearest_box = None
        for box in results.xyxy[0]:
            if box[5] != 0:
                continue

            median = self.zed.get_median_depth(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if not math.isnan(median) and not math.isinf(median) and not median <= 0:
                nearest_object = min(median, nearest_object)
                nearest_box = box
                
        return nearest_object, nearest_box
    
    def get_offset(self, box, results, image):
        if (box is None):
            return 0, 0
        
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        object_center_x = (xB + xA) / 2
        object_center_y = (yB + yA) / 2
        
        offset_x = center_x - object_center_x
        offset_y = center_y - object_center_y
        
        return offset_x, offset_y
        

    def parse_arguments(self):
        host = args.host_ip
        port = args.port
        show_boxes = args.show_boxes
        model_name = args.model_name
        get_depth = args.get_depth
        show_depth = args.show_depth

        if host is None:
            host = '192.168.194.3'

        if port is None:
            port = 8990

        if show_boxes is None or show_boxes == 'True':
            show_boxes = True
        else:
            show_boxes = False

        if get_depth is None or get_depth == 'True':
            get_depth = True
        else:
            get_depth = False

        if show_depth is None or show_depth == 'False':
            show_depth = False
        else:
            show_depth = True

        if model_name is None:
            model_name = './models_folder/yolov5m.pt'
        else:
            model_name = "./models_folder/" + model_name

        return host, port, show_boxes, model_name, get_depth, show_depth

    #creates camera objects
    def create_camera_object(self):
        zed = None
        cap = None
        #import success tests if zed sdk imported successfully
        if import_success:
            zed = Zed()
            state = zed.open()
            if state != sl.ERROR_CODE.SUCCESS:
                zed = None
                print("Zed camera not found, using webcam")
                cap = cv2.VideoCapture(0)
        else:
            zed = None
            print("camera library not found, using webcam")
            cap = cv2.VideoCapture(0)

        return zed, cap

    #send image to socket, done as a process
    def send_image_to_socket(self, socket, image):
        try:
            socket.send_video(image)
        except Exception as e:
            print(e)
            socket.client_socket.close()

    #gets image from either zed or cv2 capture
    def get_image(self, zed, cap):
        if zed is not None:
            image = zed.get_image()
        elif cap is not None:
            _, image = cap.read()
        else:
            print("No camera found, exiting")
        
        return image

    def swap_model(self, model_name):
        self.detection.load_new_model(model_name)


    def run_loop(self):
        host, port, show_boxes, model_name, get_depth, show_depth = self.parse_arguments()

        socket = Socket_Client.Client(host, port)
        socket.connect_to_server()
        self.detection = yv5.ObjDetModel(model_name)
        
        depth = 0

        #create camera objects
        self.zed, self.cap = self.create_camera_object()
        
        while True:
            image = self.get_image(self.zed, self.cap)

            #run yolo detection
            if (self.frame_count >= self.skip_frames):
                results = self.detection.detect_in_image(image)
                self.frame_count = 0
            else:
                self.frame_count += 1

            #get depth image from the zed if zed is initialized and user added the show depth argument
            if self.zed is not None and show_depth:
                image = self.zed.get_depth_image()
            #shows boxes (set to True by default)
            if show_boxes:
                image = gui_helper.draw_boxes(image, results)
                image = gui_helper.draw_lines(image, results)
            #get depth of nearest object(set to True by default)
            if self.zed is not None and get_depth:
                depth, box = self.get_nearest_object(results)
                
                self.offset_x.value, self.offset_y.value = self.get_offset(box, results, image)
                self.depth.value = float(depth)
            
            #starting imu code
            if (import_success):
                orientation, lin_acc, ang_vel = self.zed.get_imu()
                
                #set shared memory values
                self.linear_acceleration_x.value = lin_acc[0]
                self.linear_acceleration_y.value = lin_acc[1]
                self.linear_acceleration_z.value = lin_acc[2]
                self.angular_velocity_x.value = ang_vel[0]
                self.angular_velocity_y.value = ang_vel[1]
                self.angular_velocity_z.value = ang_vel[2]
                self.orientation_x.value = orientation[0]
                self.orientation_y.value = orientation[1]
                self.orientation_z.value = orientation[2]

                


            #send image over socket on another processor
            send_process = multiprocessing.Process(target = self.send_image_to_socket(socket, image))
            send_process.start()


if __name__ == '__main__':
    loop_object = VideoRunner(None)
    loop_object.run_loop()
