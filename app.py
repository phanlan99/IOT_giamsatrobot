from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import serial
import time
import threading
import json
from datetime import datetime

app = Flask(__name__)

class RobotController:
    def __init__(self):
        # Khởi tạo các biến trạng thái
        self.camera = None
        self.model = None
        self.serial_connection = None
        self.camera_connected = False
        self.model_loaded = False
        self.serial_connected = False
        self.is_running = False
        self.current_frame = None
        
        # Biến global cho serial
        self.a = 1
        self.limits = [1180, 0, 1180, 720]
        
        # Thông số camera
        self.camera_matrix = np.array([[1.50380710e+03, 0.00000000e+00, 6.86478160e+02],
                                     [0.00000000e+00, 1.50474731e+03, 4.80534599e+02],
                                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        self.distortion_coeffs = np.array([[-0.44584434, 0.30173071, 0.00053927, 0.00136141, -0.10749109]])
        
        # Thiết lập các thông số ma trận xoay và dịch tọa độ
        self.cm_to_pixel = (291.5 / 1280)
        Rot_180X = [[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]]
        rad = (-90 / 180) * np.pi
        Rot_90Z = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
        RO_C = np.dot(Rot_180X, Rot_90Z)
        dO_C = [[260+10], [-175-45], [0]]
        self.HO_C = np.concatenate((RO_C, dO_C), axis=1)
        self.HO_C = np.concatenate((self.HO_C, [[0, 0, 0, 1]]), axis=0)
        
        # Danh sách tên lớp và màu sắc
        self.classNames = ['beroca', 'cachua', 'cam', 'egg', 'maleutyl', 'probio', 'sui', 'topralsin', 'vitatrum', 'zidocinDHG']
        import random
        random.seed(42)
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in self.classNames]
        
        self.log_messages = []

    def log_message(self, message):
        """Thêm log message với timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 100:  # Giới hạn 100 messages
            self.log_messages.pop(0)
        print(log_entry)

    def connect_camera(self, camera_id=1):
        """Kết nối camera"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if self.camera.isOpened():
                self.camera_connected = True
                self.log_message(f"Camera {camera_id} connected successfully")
                return True
            else:
                self.log_message(f"Failed to connect camera {camera_id}")
                return False
        except Exception as e:
            self.log_message(f"Camera connection error: {str(e)}")
            return False

    def disconnect_camera(self):
        """Ngắt kết nối camera"""
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.camera_connected = False
            self.is_running = False
            self.log_message("Camera disconnected")
            return True
        except Exception as e:
            self.log_message(f"Camera disconnection error: {str(e)}")
            return False

    def load_model(self, model_path=r'D:\muadoan\PBL6\PBL6\Code\Python\IOT\best.pt'):
        """Tải model YOLO"""
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.log_message(f"YOLO model loaded from {model_path}")
            return True
        except Exception as e:
            self.log_message(f"Model loading error: {str(e)}")
            return False

    def unload_model(self):
        """Gỡ bỏ model"""
        self.model = None
        self.model_loaded = False
        self.log_message("YOLO model unloaded")
        return True

    def connect_serial(self, port='COM5', baudrate=9600):
        """Kết nối serial"""
        try:
            self.serial_connection = serial.Serial(port, baudrate, timeout=0.1)
            self.serial_connected = True
            self.log_message(f"Serial connected to {port} at {baudrate} baud")
            return True
        except Exception as e:
            self.log_message(f"Serial connection error: {str(e)}")
            return False

    def disconnect_serial(self):
        """Ngắt kết nối serial"""
        try:
            if self.serial_connection:
                self.serial_connection.close()
                self.serial_connection = None
            self.serial_connected = False
            self.log_message("Serial disconnected")
            return True
        except Exception as e:
            self.log_message(f"Serial disconnection error: {str(e)}")
            return False

    def transmit_data(self, data):
        """Gửi dữ liệu qua serial"""
        if self.serial_connected and self.serial_connection:
            if self.a == 1:
                try:
                    self.serial_connection.write(data.encode())
                    self.a = 0
                    self.log_message(f"Data sent: {data.strip()}")
                except Exception as e:
                    self.log_message(f"Serial transmission error: {str(e)}")

    def undistort_and_crop(self, frame):
        """Hiệu chỉnh ảnh bị biến dạng"""
        h, w = frame.shape[:2]
        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs, None)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h))
        x, y, w, h = roi
        undistorted_frame = undistorted_frame[y:y + h, x:x + w]
        return undistorted_frame

    def inverse_kinematics(self, Px_inv, Py_inv, Pz_inv=125):
        """Tính động học nghịch"""
        # Định nghĩa các tham số
        d1 = 206
        a1 = 0
        a2 = 320
        a3 = 14
        d4 = 327
        d6 = 158

        # Ma trận xoay R06
        R06 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])

        # Tính Wx, Wy, Wz
        Wx = Px_inv - d6 * R06[2, 0]
        Wy = Py_inv - d6 * R06[2, 1]
        Wz = Pz_inv - d6 * R06[2, 2]

        # Tính theta1
        theta1 = np.degrees(np.arctan2(Wy, Wx))

        # Tính các góc phi1, phi2 cho theta2
        cos_phi = ((np.sqrt(Wx ** 2 + Wy ** 2) - a1) ** 2 + (Wz - d1) ** 2 + a2 ** 2 - (a3 ** 2 + d4 ** 2)) / (
                2 * a2 * np.sqrt((np.sqrt(Wx ** 2 + Wy ** 2) - a1) ** 2 + (Wz - d1) ** 2))
        cos_phi = np.clip(cos_phi, -1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)

        phi1 = np.degrees(np.arctan2(sin_phi, cos_phi))
        phi2 = np.degrees(np.arctan2(-sin_phi, cos_phi))

        sigma = np.degrees(np.arctan2(Wz - d1, np.sqrt(Wx ** 2 + Wy ** 2) - a1))
        t2_inv1 = sigma - phi1
        t2_inv2 = sigma - phi2

        # Tính các góc gamma1, gamma2 cho theta3
        cos_gamma = (-(a2 ** 2 + a3 ** 2 + d4 ** 2) + (np.sqrt(Wx ** 2 + Wy ** 2) - a1) ** 2 + (Wz - d1) ** 2) / (
                2 * a2 * np.sqrt(a3 ** 2 + d4 ** 2))
        cos_gamma = np.clip(cos_gamma, -1, 1)
        sin_gamma = np.sqrt(1 - cos_gamma ** 2)

        gamma1 = np.degrees(np.arctan2(sin_gamma, cos_gamma))
        gamma2 = np.degrees(np.arctan2(-sin_gamma, cos_gamma))

        beta = np.degrees(np.arctan2(d4, a3))
        t3_inv1 = gamma1 + beta
        t3_inv2 = gamma2 + beta

        # Chọn giá trị nhỏ nhất (theo giá trị tuyệt đối) cho theta2 và theta3
        theta2 = t2_inv2
        theta3 = t3_inv2

        # Tính theta4 và theta6
        T03_inv = np.array([
            [np.cos(np.radians(theta2 + theta3)) * np.cos(np.radians(theta1)),
            np.cos(np.radians(theta2 + theta3)) * np.sin(np.radians(theta1)), np.sin(np.radians(theta2 + theta3))],
            [np.sin(np.radians(theta1)), -np.cos(np.radians(theta1)), 0],
            [np.sin(np.radians(theta2 + theta3)) * np.cos(np.radians(theta1)),
            np.sin(np.radians(theta2 + theta3)) * np.sin(np.radians(theta1)), -np.cos(np.radians(theta2 + theta3))]
        ])

        R36 = np.dot(T03_inv, R06)
        r13, r23, r33 = R36[0, 2], R36[1, 2], R36[2, 2]
        r31, r32 = R36[2, 0], R36[2, 1]

        # Tính theta4
        theta4_1 = np.degrees(np.arctan2(-r23, -r13))
        theta4_2 = np.degrees(np.arctan2(r23, r13))

        # Tính theta6
        theta6_1 = np.degrees(np.arctan2(r32, -r31))
        theta6_2 = np.degrees(np.arctan2(-r32, r31))

        # Tính theta5
        costheta5 = r33
        sintheta5 = np.sqrt(1 - costheta5**2)
        theta5_1 = np.degrees(np.arctan2(-sintheta5, costheta5))
        theta5_2 = np.degrees(np.arctan2(sintheta5, costheta5))

        # Chọn cặp theta4, theta5, theta6 dựa trên giá trị nhỏ nhất của theta4
        if abs(theta4_1) < abs(theta4_2):
            theta4 = round(theta4_1, 2)
            theta5 = round(theta5_1, 2)
            theta6 = round(theta6_1, 2)
        else:
            theta4 = round(theta4_2, 2)
            theta5 = round(theta5_2, 2)
            theta6 = round(theta6_2, 2)

        return theta1, theta2, theta3, theta4, theta5, theta6

    def process_frame(self):
        """Xử lý frame từ camera"""
        if not self.camera_connected or not self.camera:
            return None

        ret, frame = self.camera.read()
        if not ret:
            return None

        # Hiệu chỉnh ảnh
        undistorted_frame = self.undistort_and_crop(frame)

        if self.model_loaded and self.model:
            # Phân loại bằng YOLO
            results = self.model(undistorted_frame, stream=True, verbose=False)
            num_seeds = 0

            # Danh sách để lưu bounding box và khoảng cách
            bounding_boxes = []
            
            # Kẻ line
            cv2.line(undistorted_frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 0, 255), 5)

            for r in results:
                boxes = r.boxes
                num_seeds = len(boxes)
                for i, box in enumerate(boxes):
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Centerpoint
                    centerx = (x1 + x2) / 2
                    centery = (y1 + y2) / 2
                    centimetx = round(centerx * self.cm_to_pixel, 2)
                    centimety = round(centery * self.cm_to_pixel, 2)

                    # Tính khoảng cách từ gốc tọa độ
                    distance = math.sqrt(centimetx**2 + centimety**2)
                    bounding_boxes.append((distance, box, (centimetx, centimety), (centerx, centery)))

            # Sắp xếp bounding box theo khoảng cách
            bounding_boxes.sort(key=lambda x: x[0])

            # Vẽ bounding box và xử lý
            for idx, (distance, box, (centimetx, centimety), (centerx, centery)) in enumerate(bounding_boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confident
                conf = math.ceil((box.conf[0] * 100)) / 100

                # ClassName
                cls = int(box.cls[0])

                if 0 <= cls < len(self.classNames):
                    currentClass = self.classNames[cls]
                    myColor = self.colors[cls]

                    cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), myColor, 3)
                    cv2.circle(undistorted_frame, (int(centerx), int(centery)), radius=3, color=(255, 255, 255), thickness=5)
                    cvzone.putTextRect(undistorted_frame, f'{self.classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                                    scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                    cv2.putText(undistorted_frame, f'{idx + 1}', (max(10, x1), max(0, y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 3)

                    # Tính toạ độ thực tế
                    PC = np.array([[centimetx], [centimety], [-400], [1]])
                    P0 = np.dot(self.HO_C, PC)
                    x0 = round(P0[0, 0], 2)
                    y0 = round(P0[1, 0], 2)
                    z0 = round(P0[2, 0], 2)

                    cvzone.putTextRect(undistorted_frame, f'Toado: x={x0:.2f}, y={y0:.2f}', (x2, y2), scale=1, thickness=1, colorR=myColor)

                    # Xử lý robot cho object đầu tiên và trong vùng cho phép
                    if idx + 1 == 1 and centerx < 1180:
                        # Tính động học nghịch
                        t1_inv, t2_inv, t3_inv, t4_inv, t5_inv, t6_inv = self.inverse_kinematics(x0, y0)

                        # Làm tròn các góc
                        t1_inv_rounded = round(t1_inv, 2)
                        t2_inv_rounded = round(t2_inv, 2)
                        t3_inv_rounded = round(t3_inv, 2)
                        t4_inv_rounded = round(t4_inv, 2)
                        t5_inv_rounded = round(t5_inv, 2)
                        t6_inv_rounded = round(t6_inv, 2)

                        # Gửi dữ liệu đến Arduino qua serial
                        data = f'Start,{currentClass},{t1_inv_rounded},{t2_inv_rounded},{t3_inv_rounded},{t4_inv_rounded},{t5_inv_rounded},{t6_inv_rounded},{x0},{y0},{z0},\n'
                        self.transmit_data(data)

                        if self.serial_connected and self.serial_connection:
                            try:
                                response = self.serial_connection.readline().strip()
                                if response == b'Done':
                                    self.a = 1
                                    self.log_message("Received 'Done' from Arduino")
                            except:
                                pass

            # Thêm nhãn số lượng hạt
            label = f'Seed Count: {num_seeds}'
            cvzone.putTextRect(undistorted_frame, label, (10, 710), scale=1, thickness=1, colorR=(0, 0, 255))

        self.current_frame = undistorted_frame
        return undistorted_frame

    def generate_frames(self):
        """Generator để tạo frames cho video stream"""
        while self.is_running:
            frame = self.process_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

# Khởi tạo controller
robot_controller = RobotController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(robot_controller.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing')
def start_processing():
    robot_controller.is_running = True
    return jsonify({'status': 'success', 'message': 'Processing started'})

@app.route('/stop_processing')
def stop_processing():
    robot_controller.is_running = False
    return jsonify({'status': 'success', 'message': 'Processing stopped'})

@app.route('/connect_camera')
def connect_camera():
    camera_id = request.args.get('camera_id', 1, type=int)
    success = robot_controller.connect_camera(camera_id)
    return jsonify({'status': 'success' if success else 'error', 
                    'connected': robot_controller.camera_connected})

@app.route('/disconnect_camera')
def disconnect_camera():
    success = robot_controller.disconnect_camera()
    return jsonify({'status': 'success' if success else 'error',
                    'connected': robot_controller.camera_connected})

@app.route('/load_model')
def load_model():
    model_path = request.args.get('model_path', r'D:\muadoan\PBL6\PBL6\Code\Python\IOT\best.pt')
    success = robot_controller.load_model(model_path)
    return jsonify({'status': 'success' if success else 'error',
                    'loaded': robot_controller.model_loaded})

@app.route('/unload_model')
def unload_model():
    success = robot_controller.unload_model()
    return jsonify({'status': 'success' if success else 'error',
                    'loaded': robot_controller.model_loaded})

@app.route('/connect_serial')
def connect_serial():
    port = request.args.get('port', 'COM5')
    baudrate = request.args.get('baudrate', 9600, type=int)
    success = robot_controller.connect_serial(port, baudrate)
    return jsonify({'status': 'success' if success else 'error',
                    'connected': robot_controller.serial_connected})

@app.route('/disconnect_serial')
def disconnect_serial():
    success = robot_controller.disconnect_serial()
    return jsonify({'status': 'success' if success else 'error',
                    'connected': robot_controller.serial_connected})

@app.route('/status')
def get_status():
    return jsonify({
        'camera_connected': robot_controller.camera_connected,
        'model_loaded': robot_controller.model_loaded,
        'serial_connected': robot_controller.serial_connected,
        'is_running': robot_controller.is_running
    })

@app.route('/logs')
def get_logs():
    return jsonify({'logs': robot_controller.log_messages[-20:]})  # Lấy 20 log gần nhất

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)