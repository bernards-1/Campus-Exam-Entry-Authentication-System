README
======

Project Overview
----------------
This project is a **Face Recognition + RFID** based student attendance and identity verification system.  
Main features include:
- Student identification using YOLOv8 face recognition model
- Student verification using RFID cards
- Storing student information in an SQLite database (students.db)
- Providing a simple graphical user interface (newfacewithUI.py)
- Capturing and saving student face images



Privacy Notice  
For privacy reasons, students.db, studentPhoto/, and personPhoto/ are empty.
Please run database.py to generate a new database table, and upload your own photos to studentPhoto/ or personPhoto/.

- The `database.py` file contains some **dummy data*
- You need to upload your own photos into the `personPhoto/` folder if you want to test face recognition.
- A  `students.db` table will be created automatically when you run `database.py`. 

File Structure
--------------
### Face Recognition (Python)
- project/database.py       : Database operations
- project/newfacewithUI.py  : Main program with GUI for face recognition and management
- project/saveimage.py      : Script for capturing and saving face images
- project/students.db       : SQLite database file
- project/yolov8n-face.pt   : YOLOv8 face detection model file
- project/studentPhoto/     : Folder student ID photos one student one PIC
- project/personPhoto/      : Upload own photo At least 50

### RFID (Arduino)
- RFID/RFID.ino             : Arduino program for RFID card reading and communication


## Environment Setup
### Requirements
- **Python 3.10 or 3.11**
- **NVIDIA GPU** with CUDA 12.1 (recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt

### Hardware (ESP 32 + RFID-RC522)
Tools Required:
Arduino IDE
ESP32 development board
RC522 RFID module
USB webcam
Communication:
ESP32  Computer via Serial (USB)
Webcam  Computer via USB

How to Connect :
MFRC522        RFID Reader	
SDA	            GPIO 5	
SCK	            GPIO 18	
MOSI	            GPIO 23	
MISO	            GPIO 19	
IRQ	            Don’t connect	
GND	            GND	
RST	            GPIO 21	
3.3V	            3.3V


How to Run
----------
1. Install dependencies:
   pip install -r requirements.txt

2. Initialize the database:
   python project/database.py

3. Upload multiple photos of yourself into the `project/studentPhoto/` folder.  
   Then run:
   python project/saveimage.py
   This script will use AI to detect your face and save it into the database.

4. Update the `student_cards` table in the database to set your own RFID UID.


5. For RFID:
   - Open `RFID/RFID.ino` in Arduino IDE
   - Upload it to your Arduino board with the RFID module
   - Ensure serial communication with the Python program for attendance verification

6. Start the system with Streamlit:
   streamlit run project/newfacewithUI.py


Author
------
Developed by Bernard Seik Ping Yong Group.
