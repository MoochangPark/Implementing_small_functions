import cv2                     # 영상 처리 및 GUI 표시용
import numpy as np             # 이미지 배열 처리와 연산용
import uuid                    # 이미지 저장할 때 파일명 생성용

def rotate_image(image, angle):
    h, w = image.shape[:2]                                 # 이미지의 높이, 너비 구하기
    center = (w//2, h//2)                                  # 회전 중심 설정 (이미지 중앙)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 회전 행렬 계산
    return cv2.warpAffine(image, rotate_matrix, (w, h))          # 이미지에 회전 적용하여 반환

def blend_images(img1, img2, mode='additive'):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    def compute_luminance(img):
        return 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0] # 이미지의 각 픽셀에 대해 밝기(Luminance) 계산

    if mode == 'bright':
        lum1 = compute_luminance(img1)        # img1의 명도 계산
        lum2 = compute_luminance(img2)        # img2의 명도 계산
        mask = lum1 >= lum2        # 각 픽셀 별 명도가 더 밝은 쪽 선택
        result = np.where(mask[:, :, None], img1, img2)        # mask값을 기준으로 픽셀단위로 밝은 쪽 선택

    elif mode == 'dark':
        lum1 = compute_luminance(img1)        # img1의 명도 계산
        lum2 = compute_luminance(img2)        # img2의 명도 계산
        mask = lum1 < lum2        # 각 픽셀 별 명도가 더 어두운 쪽 선택
        result = np.where(mask[:, :, None], img1, img2)        # mask값을 기준으로 픽셀단위로 어두운 쪽 선택

    elif mode == 'additive':
        result = np.clip(img1 + img2, 0, 255)        # 두 이미지의 각 픽셀값을 더함 (범위는 0에서 255로 고정)

    elif mode == 'average':
        lum1 = compute_luminance(img1)        # img1의 명도 계산
        lum2 = compute_luminance(img2)        # img2의 명도 계산

        avg1 = np.mean(lum1)        # img1의 노출값 평균 계산
        avg2 = np.mean(lum2)        # img2의 노출값 평균 계산
        target_lum = (avg1 + avg2) / 2        # img1과 img2의 노출값의 평균 계산

        # target_lum를 각 이미지의 평균 밝기로 나눠 조정값 계산
        scale1 = target_lum / (avg1 + 1e-15)        #img1의 노출값 조절을 위한 정규화(1e-6dms 0 division 방지)
        scale2 = target_lum / (avg2 + 1e-15)        #img2의 노출값 조절을 위한 정규화(1e-6dms 0 division 방지)

        # RGB 값을 전체적으로 조정
        img1_adj = np.clip(img1 * scale1, 0, 255)
        img2_adj = np.clip(img2 * scale2, 0, 255)

        # 조정된 두 이미지를 합산
        result = np.clip((img1_adj + img2_adj) / 2, 0, 255)

    return result.astype(np.uint8)        # 다시 frame에 맞는 uint8 형식으로 변환

def main():
    capture = cv2.VideoCapture(0)          # 웹캠 열기
    if not capture.isOpened():
        print("Noting can useable camera")
        return

    step = 0                           # 현재 단계 (0: 첫 사진, 1: 두 번째 촬영, 2: 결과 보기)
    img1 = None                        # 첫 번째 사진
    angle = 0                          # 초기 회전 각도
    img2 = None                        # 두 번째 사진
    blend_mode = 'bright'           # 초기 블렌딩 모드

    while True:
        read_frame, frame = capture.read()       # 현재 프레임 읽기 (read_frame : 프레임을 읽었는지 확인(Boolean), frame : 읽은 이미지(numpy 배열))
        if not read_frame:       # 만약 프레임을 못 읽었다면 정지
            break

        display = frame.copy()        # 화면에 현재 프레임 표시하기 위해 복사

        # 화면에 띄울 프레임과 문장 처리
        if step == 0 and img1 is None:        # 현재 첫번째 촬영을 진행중이라면
            cv2.putText(display, "Press SpaceBar to capture the first shot.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 화면 상단에 촬영법 출력
        
        elif step == 0 and img1 is not None:        # 현재 첫번째 촬영을 찍었다면(spacebar를 누른 경우)
            rotated = rotate_image(img1, angle)          # 촬영한 첫 이미지 회전
            display = rotated        # 회전시킨 화면을 출력
            cv2.putText(display, f"Angle: {angle}, A/D - Rotate, Enter - Next", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 화면 상단에 회전상태 문구 출력

        elif step == 1 and img1 is not None:        # 현재 두번째 촬영을 진행중이라면
            overlay = cv2.addWeighted(frame, 0.7, img1, 0.3, 0)   # 현재 영상에 첫 이미지 오버레이
            display = overlay        # 오버레이한 img1과 프레임을 합쳐서 출력
            cv2.putText(display, "Press SpaceBar to capture the second shot.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 화면 상단에 촬영법 출력

        elif step == 2 and img1 is not None and img2 is not None:        # 현재 두번째 촬영까지 끝낸경우
            result = blend_images(img1, img2, blend_mode)        # 두 이미지 블렌딩 결과 표시
            display = result        # 블렌딩한 결과 출력
            cv2.putText(display, f"Mode: {blend_mode} [1: Bright | 2: Dark | 3: Additive | 4: Average]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 화면 상단에 모드 출력
            cv2.putText(display, "if press ESC to Save & Exit", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 화면 상단에 다음 단계방법 출력

        cv2.imshow("Fujifilm - Multi Exposure", display)         # 화면에 표시
        
        # 입력키 대기 후 동작
        key = cv2.waitKey(1)

        if key == 27:  # ESC를 눌렀을 경우
            if step == 2: # 블렌딩까지 끝난경우 저장하기 위한 것으로 판단
                filename = f"multi_exposure_{uuid.uuid4().hex[:8]}.png"   # 고유 파일명 생성
                result = blend_images(img1, img2, blend_mode)             # 블렌딩 결과 저장
                cv2.imwrite(filename, result) # 이미지 저장
                print(f"Complete Save as : {filename}")
            break

        if step == 0: # 첫 번째 사진 촬영 단계 
            if key == ord(' '):               # 스페이스바: 첫 사진 촬영
                img1 = frame.copy()
                print(img1)
            elif key == ord('a'):             # A: 왼쪽으로 회전
                angle -= 2
            elif key == ord('d'):             # D: 오른쪽으로 회전
                angle += 2
            elif key == 13 and img1 is not None:  # Enter: 회전 적용 후 다음 단계
                img1 = rotate_image(img1, angle)        # 첫번째 이미지 배열 저장
                step = 1 # 다음 단계로 이동

        elif step == 1: # 두 번째 사진 촬영 단계
            if key == ord(' '):               # 스페이스바: 두 번째 사진 촬영
                img2 = frame.copy()        # 두번째 이미지 배열 저장
                step = 2 # 다음 단계로 이동

        elif step == 2: # 블렌딩 모드 선택 단계
            if key == ord('1'):
                blend_mode = 'bright'
            elif key == ord('2'):
                blend_mode = 'dark'
            elif key == ord('3'):
                blend_mode = 'additive'
            elif key == ord('4'):
                blend_mode = 'average'

    capture.release()               # 웹캠 해제
    cv2.destroyAllWindows()     # 모든 창 닫기

if __name__ == '__main__':
    main()                      # 프로그램 실행
