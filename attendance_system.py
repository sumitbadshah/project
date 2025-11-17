import sys
import cv2
import numpy as np
import sqlite3
import datetime
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLineEdit,
                             QMessageBox, QListWidget, QDialog, QTextEdit)

class SimpleFaceRecognizer:
    def __init__(self):
        self.face_data = []
        self.labels = []
        self.label_names = {}
        
    def train(self, faces, labels, label_names):
        self.face_data = faces
        self.labels = labels
        self.label_names = label_names
        
    def predict(self, face_roi):
        if not self.face_data:
            return -1, 100  
        
        
        best_match_id = -1
        best_similarity = float('inf')
        
        for i, trained_face in enumerate(self.face_data):
            
            if trained_face.shape != face_roi.shape:
                resized_face = cv2.resize(face_roi, (trained_face.shape[1], trained_face.shape[0]))
            else:
                resized_face = face_roi
                
            
            similarity = np.mean((trained_face.astype(float) - resized_face.astype(float)) ** 2)
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match_id = self.labels[i]
        
        
        confidence = max(0, min(100, best_similarity / 10))
        
        return best_match_id, confidence

class FaceRegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register New Face")
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name")
        layout.addWidget(self.name_input)
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Enter ID (optional)")
        layout.addWidget(self.id_input)
        
        button_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Face")
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        self.capture_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class AttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Attendance System")
        self.setGeometry(100, 100, 1000, 700)
        
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = SimpleFaceRecognizer()
        
        
        self.init_database()
        
        
        self.model_trained = False
        self.load_trained_model()
        
        self.known_face_ids = []
        self.known_face_names = []
        
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_capturing = False
        
        self.current_frame = None
        self.attendance_marked = set()
        
        self.init_ui()
        self.load_known_faces()
        
    def init_database(self):
        self.conn = sqlite3.connect('attendance.db')
        self.cursor = self.conn.cursor()
        
    
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_id TEXT,
                image BLOB
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT,
                name TEXT,
                timestamp DATETIME
            )
        ''')
        
        self.conn.commit()
    
    def load_known_faces(self):
        self.cursor.execute("SELECT face_id, name FROM faces")
        faces = self.cursor.fetchall()
        self.known_face_ids = [face[0] for face in faces]
        self.known_face_names = [face[1] for face in faces]
    
    def load_trained_model(self):
        try:
            # Load face data from database
            self.cursor.execute("SELECT face_id, image FROM faces")
            faces = self.cursor.fetchall()
            
            if not faces:
                return
                
            face_samples = []
            labels = []
            label_names = {}
            
            for face in faces:
                face_id = face[0]
                img_data = face[1]
                
                
                nparr = np.frombuffer(img_data, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                face_samples.append(img_np)
                labels.append(face_id)  # Store the ID as string, not integer
                
            
                self.cursor.execute("SELECT name FROM faces WHERE face_id = ?", (face_id,))
                name_result = self.cursor.fetchone()
                if name_result:
                    label_names[face_id] = name_result[0]  # Use string ID as key
            
            
            self.recognizer.train(face_samples, labels, label_names)
            self.model_trained = True
            
        except Exception as e:
            print(f"Error loading trained model: {e}")
            self.model_trained = False
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout()
        central_widget.setLayout(layout)
        
        
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        
        if not self.cap.isOpened():
            self.video_label.setText("Webcam not available\nPlease check your camera connection")
        else:
            self.video_label.setText("Webcam Feed\nClick Start to begin")
            
        left_layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Status: Not started")
        left_layout.addWidget(self.status_label)
        
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.register_btn = QPushButton("Register Face")
        self.train_btn = QPushButton("Train Model")
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.register_btn)
        control_layout.addWidget(self.train_btn)
        
        right_layout.addLayout(control_layout)
        
        
        right_layout.addWidget(QLabel("Attendance Log:"))
        self.attendance_log = QTextEdit()
        self.attendance_log.setReadOnly(True)
        right_layout.addWidget(self.attendance_log)
        
        
        self.view_records_btn = QPushButton("View Attendance Records")
        right_layout.addWidget(self.view_records_btn)
        
        layout.addWidget(left_widget)
        layout.addWidget(right_widget)
        
        
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.register_btn.clicked.connect(self.register_face)
        self.train_btn.clicked.connect(self.train_model)
        self.view_records_btn.clicked.connect(self.view_attendance_records)
        
    
        self.stop_btn.setEnabled(False)
        
        
        if not self.cap.isOpened():
            self.start_btn.setEnabled(False)
            self.register_btn.setEnabled(False)
        
    def start_capture(self):
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Webcam is not available")
            return
            
        self.is_capturing = True
        self.timer.start(20)  # Update every 20ms
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running")
        
    def stop_capture(self):
        self.is_capturing = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
    
            self.current_frame = frame.copy()
            
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
          
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
           
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if self.model_trained:
                    # Recognize the face
                    id_, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # If confidence is less than 70, consider it a match
                    if confidence < 70 and id_ != -1:
                        name = self.recognizer.label_names.get(id_, "Unknown")
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        
                        # Mark attendance if not already marked today
                        today = datetime.date.today().strftime("%Y-%m-%d")
                        if f"{name}_{today}" not in self.attendance_marked:
                            self.mark_attendance(str(id_), name)
                            self.attendance_marked.add(f"{name}_{today}")
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Model not trained", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def mark_attendance(self, face_id, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        self.cursor.execute("INSERT INTO attendance (face_id, name, timestamp) VALUES (?, ?, ?)",
                           (face_id, name, timestamp))
        self.conn.commit()
        
        
        self.attendance_log.append(f"{timestamp} - {name} marked present")
    
    def register_face(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Please start the camera first")
            return
            
        dialog = FaceRegistrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.name_input.text().strip()
            face_id = dialog.id_input.text().strip()
            
            if not name:
                QMessageBox.warning(self, "Warning", "Please enter a name")
                return
                
            if not face_id:
                face_id = str(len(self.known_face_ids) + 1)
            
            
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                QMessageBox.warning(self, "Warning", "No face detected in the current frame")
                return
                
           
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
          
            ret, buffer = cv2.imencode('.jpg', face_roi)
            if ret:
                
                face_blob = buffer.tobytes()
                
                
                self.cursor.execute("INSERT INTO faces (name, face_id, image) VALUES (?, ?, ?)",
                                   (name, face_id, face_blob))
                self.conn.commit()
                
                QMessageBox.information(self, "Success", f"Face registered for {name} with ID {face_id}")
                
                
                self.load_known_faces()
            else:
                QMessageBox.warning(self, "Error", "Failed to process face image")
    
    def train_model(self):
        
        self.cursor.execute("SELECT face_id, image FROM faces")
        faces = self.cursor.fetchall()
        
        if not faces:
            QMessageBox.warning(self, "Warning", "No faces registered in the database")
            return
            
        face_samples = []
        labels = []
        label_names = {}
        
        for face in faces:
            face_id = face[0]
            img_data = face[1]
            
           
            nparr = np.frombuffer(img_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            face_samples.append(img_np)
            labels.append(face_id)  
            
            
            self.cursor.execute("SELECT name FROM faces WHERE face_id = ?", (face_id,))
            name_result = self.cursor.fetchone()
            if name_result:
                label_names[face_id] = name_result[0]  # Use string ID as key
        
        
        self.recognizer.train(face_samples, labels, label_names)
        self.model_trained = True
        
        QMessageBox.information(self, "Success", f"Model trained with {len(face_samples)} face samples")
    
    def view_attendance_records(self):
        records_dialog = QDialog(self)
        records_dialog.setWindowTitle("Attendance Records")
        records_dialog.setModal(True)
        records_dialog.setFixedSize(600, 400)
        
        layout = QVBoxLayout()
        
     
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Name", "Date", "Time"])
        
        
        self.cursor.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC")
        records = self.cursor.fetchall()
        
        table.setRowCount(len(records))
        for row, record in enumerate(records):
            name, timestamp = record
            date, time = timestamp.split(' ')
            
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(date))
            table.setItem(row, 2, QTableWidgetItem(time))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
       
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(records_dialog.accept)
        layout.addWidget(close_btn)
        
        records_dialog.setLayout(layout)
        records_dialog.exec_()
    
    def closeEvent(self, event):
        
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceSystem()
    window.show()
    sys.exit(app.exec_())
