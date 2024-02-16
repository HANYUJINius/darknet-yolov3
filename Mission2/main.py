import sys, os
# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/ddwu/opencv/opencv-4.4.0/build/darknet/python'))
import cv2
import darknet as dn

# darknet 라이브러리 설정
dn.set_gpu(0)   #GPU  사용 설정
net = dn.load_net(b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/yolov3.cfg", b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/yolov3.weights", 0)
meta = dn.load_meta(b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/coco.data")


# OpenCV 설정
cap = cv2.VideoCapture(0, cv2.CAP_V4L)   # 웹캠을 이용하여 비디오를 캡처
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

if cap.isOpened():                                  # 비디오 캡처 객체 초기화 확인
    file_path = '/home/ddwu/Desktop/mission2.mp4'   # 저장할 비디오 파일 경로
    fps = 10                                        # 비디오 프레임 속도
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # 비디오 인코딩 포맷 문자
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
    size = (width, height)                              # 프레임 크기
    
    out = cv2.VideoWriter(file_path, fourcc, fps, size)  # 비디오 녹화 객체 생성

    while True:
        ret, frame = cap.read()                     # 비디오 프레임 읽기

        if ret:                                     # 프레임 읽기가 성공한 경우
            # Darknet에서 객체 감지
            r = dn.detect(net, meta, frame)
            #print(r)                               # 감지된 객체 정보 출력

            for detection in r:
                label = detection[0]
                confidence = detection[1]
                bbox = detection[2]

                left, top, right, bottom = bbox
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)

                # 사각형 그리기
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # 레이블과 신뢰도 표시
                label_text = f"{label.decode('utf-8')}: {confidence:.2f}"
                cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 비디오 플레이어 창에 프레임 표시
            cv2.imshow('mission2', frame)
            out.write(frame)                        # 비디오 파일에 프레임 저장

            if cv2.waitKey(int(1000/fps)) != -1:    # 키 입력을 기다림
                break                               # 아무 키나 누르면 프로그램 종료

        else:
            print(b'no file!')
            break

    out.release()                                   # 녹화 종료
else:
    print(b"Can't open camera!")

cap.release()                                       # 비디오 캡처 객체 해제
cv2.destroyAllWindows()                             # 모든 창 닫기
