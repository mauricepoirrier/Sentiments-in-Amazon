import numpy as np
import cv2
import boto3
import datetime
import base64



VIDEO_SOURCE = '/dev/video0'
WINDOW_NAME  = 'sentiments'
FRAME_SKIP   = 7
BUCKET_NAME  = 'emotions-photos'
PHOTO_PATH   = 'photos/'


def open_cam(uri, width=1280, height=720, latency=2000):
    gst_str = ('v4l2src do-timestamp=TRUE device={} ! videoconvert ! '
               'h264parse ! omxh264dec !'
               'video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def main():
    rek = boto3.client('rekognition')

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    cap.set(5,60)
    if not cap.isOpened():
        sys.exit('Failed to open video file!')
    #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    #cv2.setWindowTitle(WINDOW_NAME, 'Sentiment Analysis video test')

    _frame_number = 0
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            print('VidepCapture.read() failed. Exiting...')
            break

        _frame_number += 1
        if _frame_number % FRAME_SKIP != 0:
            continue
        #cv2.imshow(WINDOW_NAME, frame)
        retval, buffer = cv2.imencode('.jpg', frame)
        base_64_image = base64.b64encode(buffer)
        base_64_binary = base64.decodebytes(base_64_image)
        response = rek.detect_faces(Image={'Bytes': base_64_binary},Attributes=['ALL'])
        max_val=0
        max_type=''
        for obj in response['FaceDetails'][0]['Emotions']:
            if obj['Confidence'] > max_val:
                max_val = obj['Confidence']
                max_type = obj['Type'] 
        print('Type: {}, Confidence: {}'.format(max_type,max_val))
        print('Gender: {}, Confidence: {}'.format(response['FaceDetails'][0]['Gender']['Value'],response['FaceDetails'][0]['Gender']['Confidence'])) 
        print('Smile: {}, Confidence: {}'.format(response['FaceDetails'][0]['Smile']['Value'],response['FaceDetails'][0]['Smile']['Confidence']))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
