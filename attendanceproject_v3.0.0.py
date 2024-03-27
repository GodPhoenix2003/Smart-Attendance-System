import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import csv
from email.message import EmailMessage
import ssl
import smtplib

studNames = []
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
email_list = []
process_current_frame = True


def send_mail(recieve):
    email_sender = 'sagnikchatterjee2003.official@gmail.com'
    email_password = 'qwtthafcsfqcvwuc'
    email_receiver = recieve
    now = datetime.now()

    subject = 'Attendance Recorded'
    body = f"""
        Date: {date.today()}
        Time: {now.strftime('%H:%M:%S')}
        """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


def findEncodings():
    for image in os.listdir('C:/Smart Attendance System/Student DataBase'):
        face_image = face_recognition.load_image_file(f'C:/Smart Attendance System/Student DataBase/{image}')
        face_encoding = face_recognition.face_encodings(face_image)[0]

        known_face_encodings.append(face_encoding)
        studNames.append(os.path.splitext(image)[0])


def markAttendance(stud_name, time):
    dir = 'C:/Smart Attendance System/Attendance/'
    filename = str(date.today())
    dir = os.path.join(dir, filename)
    filename = filename + '-' + time + '.csv'
    dir = os.path.join(dir, filename)

    with open(dir, 'r+') as f:
        myStudList = f.readlines()
        studNameList = []
        for line in myStudList:
            entry = line.split(',')
            studNameList.append(entry[0])
        if name not in studNameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M:%S')
            f.writelines(f"\n{stud_name},{timestr}")
            f.close()
            # extracting name, email from csv file
            stud_email = []
            email_loc = 'C:/Smart Attendance System/email.csv'
            with open(email_loc) as em:
                email_data = csv.reader(em, delimiter=',')
                next(email_data)  # skips header in csv(important)
                for row in email_data:
                    stud_email.append(row)
            for row in stud_email:
                if stud_name == row[0]:
                    send_mail(row[1])
                    break


findEncodings()

stud = cv2.VideoCapture(0)

dir = 'C:/Smart Attendance System/Attendance/'

if os.path.exists(dir) == False:
    os.makedirs(dir)

filename = str(date.today())
dir = os.path.join(dir, filename)

if os.path.exists(dir) == False:
    os.makedirs(dir)

col_header = ["Name", "Time"]

time = datetime.now().strftime('%H-%M-%S')
filename = filename + '-' + time
filename = filename + '.csv'

with open(os.path.join(dir, filename), "w") as f:
    studList = csv.writer(f)
    studList.writerow(col_header)
    f.close()

while True:
    success, frame = stud.read()

    if process_current_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matchers = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matchers[best_match_index]:
                name = studNames[best_match_index]

            face_names.append(f'{name}')

    process_current_frame = not process_current_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name != 'Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            markAttendance(name, time)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Attendance Window', frame)

    if cv2.waitKey(1) == ord('`'):
        break


stud.release()
cv2.destroyAllWindows()

