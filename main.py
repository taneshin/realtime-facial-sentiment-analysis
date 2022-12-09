import cv2
import dlib
from deepface import DeepFace

def face_detection_yunet(image):
    import cv2
    import dlib
    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (0,0))
    h, w, _ = image.shape
    face_detector.setInputSize((w,h))
    _, faces = face_detector.detect(image)
    return [dlib.rectangle(face[0], int(face[1] * 1.15), int((face[0]+face[2]) * 1.05), face[1]+face[3]) for face in faces]

def face_clean(image):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    shape_predictor = dlib.shape_predictor("algorithm/shape_predictor_68_face_landmarks.dat")
    face_box = face_detection_yunet(image)
    shape = shape_predictor(image, face_box)
    face_chip = dlib.get_face_chip(image, shape)   
    return face_chip

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        result, image = video_capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        faces = face_clean(image)

        for face in faces:
            result = DeepFace.analyze(face , actions = ['emotion'])
            cv2.putText(face, result['dominant_emotion'], (50,390), font, 3, (0,0,255), 2, cv2.LINE_4)

    video_capture.release()
    cv2.destroyAllWindows()
