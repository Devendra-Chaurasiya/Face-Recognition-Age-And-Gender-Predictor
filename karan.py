import os
import cv2

def faceBox(faceNet, frame):
    if frame is None:
        return None, []

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Specify absolute file paths for model files
script_dir = os.path.dirname(os.path.abspath(__file__))
faceProto = os.path.join(script_dir,r"C:\Users\91914\Desktop\Face_Recognition_System\karan\opencv_face_detector.pbtxt")
faceModel = os.path.join(script_dir, r"C:\Users\91914\Desktop\Face_Recognition_System\karan\opencv_face_detector_uint8.pb")
ageProto = os.path.join(script_dir, r"C:\Users\91914\Desktop\Face_Recognition_System\karan\gender_deploy.prototxt")
ageModel = os.path.join(script_dir, r"C:\Users\91914\Desktop\Face_Recognition_System\karan\age_net.caffemodel")
genderProto = os.path.join(script_dir, r"C:\Users\91914\Desktop\Face_Recognition_System\karan\gender_deploy.prototxt")
genderModel = os.path.join(script_dir, r"C:\Users\91914\Desktop\Face_Recognition_System\karan\gender_net.caffemodel")

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture('4.mp4')

padding = 20

# Inside the while loop:
while True:
    ret, frame = video.read()

    frameFace, bboxes = faceBox(faceNet, frame)

    if frameFace is not None:
        for bbox in bboxes:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = "{},{}".format(gender, age)
            cv2.rectangle(frameFace, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age-Gender", frameFace)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break


video.release()
cv2.destroyAllWindows()