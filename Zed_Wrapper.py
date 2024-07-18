import pyzed.sl as sl
import cv2
import copy
import statistics

class Zed:

    '''
        discord: @kialli
        github: @kchan5071
        
        wrapper class for zed camera
        
        usage:
        initialize
        open camera
        
        then can get image, imu data, and depth image
        
        can also get median depth of a rectangle(4 points) in the image
        
        also has main function to test zed camera functionality
        
    '''
    
    def __init__(self):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 60
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.tracking_parameters = sl.PositionalTrackingParameters()
        self.tracking_parameters.enable_imu_fusion = True
        self.runtime_parameters = sl.RuntimeParameters()
        #self.err = self.zed.enable_positional_tracking(self.tracking_parameters)
        self.py_translation = sl.Translation()

    #opens camera, returns state to be compared against sl.ERROR_CODE.SUCCESS
    def open(self):
        state = self.zed.open(self.init_params)
        return state

    #get color image from zed camera
    def get_image(self):
        image_zed = sl.Mat()
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image_zed, sl.VIEW.RIGHT)
            return copy.deepcopy(image_zed.get_data())

    #get imu data
    def get_imu(self):
        sensors_data = sl.SensorsData()
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)
            quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
            #print("IMU Orientation: {}".format(quaternion))
            linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
            #print("IMU Acceleration: {} [m/sec^2]".format(linear_acceleration))
            angular_velocity = sensors_data.get_imu_data().get_angular_velocity()
            #print("IMU Angular Velocity: {} [deg/sec]".format(angular_velocity))

        return quaternion, linear_acceleration, angular_velocity

    #get depth image from zed camera
    def get_depth_image(self):
        image_zed = sl.Mat()
        if (self.zed.grab() == sl.ERROR_CODE.SUCCESS):
            self.zed.retrieve_image(image_zed, sl.VIEW.DEPTH)
            image = image_zed.get_data()
            return copy.deepcopy(image)
        
    #get the median depth of all points within bounds of x1, y1, x2, y2
    def get_median_depth(self, x1, y1, x2, y2):
        cam_info = self.zed.get_camera_information()
        width = cam_info.camera_configuration.resolution.width
        height = cam_info.camera_configuration.resolution.height

        #bounds check and success check
        if x1 > width or x2 > width:
            return -1
        elif y1 > height or y2 > height:
            return -1
        elif self.zed.grab() != sl.ERROR_CODE.SUCCESS :
            return -1
        
        #create depth camera object
        depth_zed = sl.Mat(cam_info.camera_configuration.resolution.width, 
                            cam_info.camera_configuration.resolution.height, 
                            sl.MAT_TYPE.F32_C1)
        
        # Retrieve depth data float 32
        self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        #take 5 sample points and return the median of them
        depth = [None] * 5
        _, depth[0] = depth_zed.get_value((x1 + x2) // 2, (y1 + y2) // 2)
        _, depth[1] = depth_zed.get_value((x1 + x2) // 4, (y1 + y2) // 2)
        _, depth[2] = depth_zed.get_value((x1 + x2) // 2, (y1 + y2) // 4)
        _, depth[3] = depth_zed.get_value(3 * (x1 + x2) // 4, (y1 + y2) // 2)
        _, depth[4] = depth_zed.get_value((x1 + x2) // 2, 3 * (y1 + y2) // 4)

        median = statistics.median(depth)

        return median
    

        
def main():
    zed = Zed()
    state = zed.open()
    while True:
        image = zed.get_depth_image()
        if (image is not None):
            cv2.imshow("image_test", image)
            cv2.waitKey(1)
        continue

if __name__ == '__main__':
    main()

