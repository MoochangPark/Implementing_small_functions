# 1. Fujifilm의 Multiple_Exposure 따라해보기

최근에 카메라에 대해 관심을 가지고 있다. 요즘 휴대폰도 기능이 좋지만 아무래도 옛날 느낌의 디카가 더 감성적이고 좋은 것 같다는 생각이 자꾸만 든다. 그러던 도중에 인☆에서 흥미로운걸 발견했다.

[fujifilmx_ca x100v 으로 멋진 사진 찍기](https://www.instagram.com/reel/DLSD6rHzhQI/?igsh=MW8zYmx6bWM1bXBzeQ==)

그래서 이런 기능은 뭐라고 하는지 찾아보니 Multiple_Exposure이라는 옛 카메라의 기법이란다. 이걸 잘하면 이런 사진들도 가능하다고 한다.

- [Multiple Exposure 이해하기](https://www.fujifilm-x.com/en-gb/learning-centre/understanding-multiple-exposure-photography/)
- [Multiple Exposure 기법의 사진찍기](https://www.leonardomascaro.com/blog/2019/7/22/in-camera-multiple-exposure-with-the-fujifilm-x-t2)
- [Multiple Exposure에서 Additive와 Average에 대하여](https://fujixweekly.com/tag/multiple-exposure/)

그래서 만들어보려고 한다.

---

### ✅ Level One - 이해하기
 📍 우선 fujifilm은 어떻게 구현하려고 하는지 GPT 시켜서 정리시켜 봤다.

```
Fujifilm X-T2 다중노출(Multiple Exposure) 기능 개요

후지필름 X-T2는 두 장 이상의 촬영 이미지를 한 장으로 겹쳐 저장하는 다중노출(Multiple Exposure) 기능을 지원한다. 이 모드에서는 사용자가 첫 번째 이미지를 촬영한 후 화면에 반투명으로 겹쳐 보여주며 두 번째 촬영을 진행하도록 유도한다. 실제 작동 과정은 다음과 같다. 먼저 드라이브 다이얼을 다중노출 모드(j)에 맞추고 첫 이미지를 촬영한다. 그 후 〈MENU/OK〉 버튼을 누르면 첫 번째 촬영 이미지가 뷰파인더(LCD/EVF)에 투명하게 중첩되어 표시된다. 이를 참고하여 두 번째 이미지를 촬영하면 카메라는 두 장의 이미지를 선택된 합성 모드로 자동 결합하여 최종 JPEG 파일로 저장한다. 예를 들어 아래 이미지는 두 인물 이미지를 겹쳐 만든 다중노출 결과의 예시이다.
다중노출 예시: 두 장의 촬영 이미지를 특정 합성 방식으로 합쳐 생성한 결과 (밝은 배경이 제거되고 피사체만 강조됨).


다중노출 합성 모드(Blend Modes)

X-T2 다중노출 모드는 합성 방식에 따라 네 가지 블렌드 모드를 제공한다. 각 모드는 두 이미지를 픽셀 단위로 조합하는 방법을 정의한다.
• Additive (덧셈): 두 이미지의 픽셀 밝기 값을 단순 합산한다. 노출을 많이 할수록 밝기가 증가하므로 노출 보정이 필요할 수 있다.
• Average (평균): 두 이미지를 평균 처리하여 노출을 자동 최적화한다. 동일 구도로 반복 촬영한 경우, 배경 노출이 균형 있게 조절된다.
• Bright (밝은 픽셀 선택): 각 위치에서 두 이미지 중 밝기가 더 큰 픽셀만을 선택한다. 즉, 두 장의 밝기 중 더 밝은 값을 취한다.
• Dark (어두운 픽셀 선택): 각 위치에서 두 이미지 중 밝기가 더 낮은(더 어두운) 픽셀을 선택한다. 즉, 두 장의 값 중 더 어두운 값을 취한다.

이들 모드는 픽셀 단위 비교 연산을 수행한다. 예를 들어 Bright 모드는 채널별 최대값(max)을, Dark 모드는 채널별 최소값(min)을 취하는 방식으로 작동한다. 이러한 방식 덕분에 다중노출 합성 시 두 이미지가 단순한 투명 중첩이 아니라 대비가 강조된 결과가 나온다.


배경 제거 및 피사체 강조 알고리즘

특히 Dark 모드를 사용하면 밝은 배경이 자동으로 제거되고 어두운 피사체가 남게 된다. Howard Grill의 설명처럼, Darken 블렌드 모드에서는 픽셀 비교 시 더 밝은(명도 높은) 픽셀을 제거하고 어두운 픽셀이 남는다. 즉, 두 이미지의 같은 위치에서 한 쪽이 밝고 다른 쪽이 어두울 때, 밝은 픽셀은 지워지고 어두운 픽셀만 최종 이미지에 남는다. 반대로 Bright 모드는 두 이미지의 밝은 영역을 살리고 어두운 부분을 제거하므로, 어두운 배경은 사라지고 밝은 실루엣만 강조된다. 이 과정에서 픽셀별 채널 값을 비교하므로, 결과적으로 밝은 배경은 퇴색하고 피사체의 선명한 실루엣이나 색상만 남게 된다. 예를 들어 아래 이미지처럼 한 장에 밝은 하늘 배경과 피사체가, 다른 장에 어두운 배경과 동일 피사체가 있을 때, Dark 모드를 쓰면 밝은 하늘 부분은 어두운 피사체로 대체되어 사라진다.
Dark 모드를 적용한 다중노출 예시: 밝은 부분(하늘)이 제거되고 어두운 인물 윤곽만 강조된다.
이러한 픽셀 비교 알고리즘은 색상의 혼합을 유발하기도 한다. 예를 들어 두 이미지에서 한쪽 픽셀은 빨강색이고 다른 쪽 픽셀은 파랑색이라면, Bright/Dark 모드에서 밝기나 색상에 따라 각 채널 값이 선택되므로 결과 색상이 변할 수 있다. 그러나 중요한 점은, 결과 이미지가 원본 이미지들의 단순한 반투명 중첩이 아닌 픽셀 값의 선택으로 이루어진다는 점이다. 이 때문에 배경처럼 불필요하게 밝은(또는 어두운) 영역은 날아가고, 중첩된 피사체가 또렷하게 드러난다.
소프트웨어로 구현하기 (OpenCV, Photoshop 등)

X-T2의 다중노출 효과는 소프트웨어에서도 블렌드 모드로 쉽게 흉내낼 수 있다. 대표적으로 OpenCV, Photoshop, Affinity 등의 도구가 있다. 예를 들어 OpenCV (Python)에서는 다음과 같이 구현한다:
• Lighten/Bright (최댓값 선택): cv2.max(img1, img2) 함수를 사용해 두 이미지의 픽셀별 최대값을 계산한다. Astro Club IITK 예제처럼 star_trail = cv2.max(star_trail, frame)는 Lighten 모드를 구현한다.
• Darken (최솟값 선택): cv2.min(img1, img2) 함수를 사용해 픽셀별 최소값을 취한다. Bright 모드와 반대로 동작하여 밝은 픽셀을 어두운 픽셀로 대체한다.
• Additive (덧셈 합성): cv2.add(img1, img2)로 픽셀 값을 더한다. 이때 결과가 최대값(255)을 넘으면 자동 클리핑된다.
• Average (평균 합성): cv2.addWeighted(img1, 0.5, img2, 0.5, 0) 등으로 두 이미지를 동일 가중치로 합산하여 평균 낸다.
이러한 함수들을 활용하면, X-T2의 각 모드와 유사한 결과를 얻을 수 있다.

예를 들어:
result_add = cv2.add(img1, img2)
result_avg = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
result_bright = cv2.max(img1, img2)    # Bright 모드
result_dark   = cv2.min(img1, img2)    # Dark 모드

위 코드는 각각 Additive, Average, Bright, Dark 모드를 구현한다.
마찬가지로 Adobe Photoshop이나 Affinity Photo 등에서도 유사한 블렌드 모드를 제공한다. Photoshop의 Lighten 모드는 채널별 최대값을, Darken 모드는 최소값을 취하는 방식이며, Linear Dodge (Add) 모드는 픽셀 밝기 덧셈(Additive)과 대응된다. Affinity Photo 역시 Lighten, Darken, Additive 등의 모드를 지원한다. 레이어 불투명도 50%를 사용하면 평균(Average) 효과를 근사할 수 있다. 이처럼 주류 이미지 편집 툴에서도 X-T2 다중노출과 동일한 픽셀 연산을 재현할 수 있다.
결론
결론적으로, Fujifilm X-T2 다중노출 모드는 촬영한 이미지를 픽셀 단위로 비교·선택하는 방식으로 합성한다. 덧셈(Additive)과 평균(Average)은 밝기 합산/평균화로, 밝은 픽셀 선택(Bright)과 어두운 픽셀 선택(Dark)은 픽셀별 최대/최소값 연산으로 처리된다. 이 중 Dark 모드의 경우 밝은 배경보다 어두운 피사체가 우선 선택되므로, 밝은 영역이 자연스럽게 삭제되고 피사체 윤곽만 남는다. 동일 원리가 OpenCV나 Photoshop 등의 소프트웨어 블렌드 모드에서도 적용되므로, 예를 들어 cv2.max(), cv2.min() 함수나 Photoshop의 Lighten/Darken 모드를 사용하면 유사한 결과를 재현할 수 있다. 이러한 내부 알고리즘 덕분에 후지필름 카메라의 다중노출 기능은 번들거리는 배경 없이 주제만 선명하게 겹치는 특별한 이미지를 만들어낸다.
```

이제 어떤식으로 작동하는지 조금 이해가 간다.
그럼 이거를 python으로 한번 구현해보려고 한다.

---

### ✅ Level Two - 개발하기

📍 우선 만들어야 한 기능은 다음과 같다.

1. 첫 번째 사진을 촬영하기
2. 첫 번째 사진의 각도 조정하기
3. 두 번째 사진을 촬영하기
4. 모드에 따라 합성하기

📍 그 다음 고려해야 할 점은 무엇일까.

1. 첫 번째 사진을 촬영하기
   - 스페이스바를 눌러서 사진 찍기
2. 첫 번째 사진의 각도 조정하기
   - A,D키를 이용해서 회전할 수 있도록 하기 (노트북을 돌릴 수는 없으니까...)
3. 두 번째 사진을 촬영하기
   - 첫 번째 사진이 오버레이 되어서 어떤 식으로 겹칠건지 미리보기
4. 모드에 따라 합성하기
   - additive, average, bright, dark 선택할 수 있도록 하기
   - 결과물 저장하기

이제 절차는 알았으니 한 번 OpenCV의 사용법을 찾아가면서 한 번 코드를 작성해보자.

---

### ✅ Level Three - 코드 작성하기

1. 코드에 사용되는 라이브러리는 다음과 같다.

| 라이브러리 | 함수 | 함수 설명 |
|--|--|--|
| OpenCV (`cv2`) | `rotate_image()` | 입력 이미지에 지정한 각도로 회전 적용 |
| OpenCV (`cv2`) | `cv2.VideoCapture()` | 웹캠(카메라) 열기 및 영상 프레임 캡처 |
| OpenCV (`cv2`) | `read()` | 웹캠에서 프레임 읽기 |
| OpenCV (`cv2`) | `cv2.getRotationMatrix2D()` | 회전 행렬 계산 (2D 회전) |
| OpenCV (`cv2`) | `cv2.warpAffine()` | 회전 행렬을 실제 이미지에 적용 |
| OpenCV (`cv2`) | `cv2.addWeighted()` | 두 이미지를 투명도 비율로 합성 |
| OpenCV (`cv2`) | `cv2.putText()` | 이미지 위에 텍스트 출력 |
| OpenCV (`cv2`) | `cv2.imshow()` | 이미지 또는 프레임을 창에 표시 |
| OpenCV (`cv2`) | `cv2.waitKey()` | 키보드 입력 대기 및 감지 |
| OpenCV (`cv2`) | `cv2.imwrite()` | 이미지 파일로 저장 |
| OpenCV (`cv2`) | `cv2.destroyAllWindows()` | 모든 OpenCV 창 닫기 |
| OpenCV (`cv2`) | `release()` | 웹캠(카메라) 자원 해제 |
| NumPy (`np`) | `blend_images()` | 이미지 블렌딩 (다중 노출) 기능 수행 |
| NumPy (`np`) | `astype()` | 데이터 타입 변환 (float32, uint8) |
| NumPy (`np`) | `np.mean()` | 배열의 평균값 계산 (밝기 평균 등) |
| NumPy (`np`) | `np.clip()` | 픽셀 값 범위 제한 (0~255) |
| NumPy (`np`) | `np.where()` | 조건에 따라 값 선택 (픽셀 단위) |
| NumPy (`np`) | `np.abs()` | 절대값 계산 (밝기 차이 비교) |
| Python 표준 | `uuid.uuid4()` | 고유한 ID를 생성하여 이미지 파일명으로 사용 |
| Python 표준 | `print()` | 디버그 또는 안내 메시지 출력 |
| Python 표준 | `__name__ == '__main__'` | 파일 직접 실행 시 `main()` 함수 실행 |

2. 중요한 Multiple_Exposure 함수 mode 구현하기
   - bright 모드
     - 각 픽셀마다 명도를 구하고, 두 이미지 중에서 밝은 쪽의 픽셀만 남겨둔다.
   - dark 모드
     - 각 픽셀마다 명도를 구하고, 두 이미지 중에서 어두운 쪽의 픽셀만 남겨둔다.
   - additive 모드
     - 두 이미지의 각 픽셀 값을 그대로 더한다. (두 이미지의 노출값이 처음부터 매우 낮아야 결과값이 좋다.)
   - average 모드
     - 두 이미지의 노출 값의 평균을 구해서 해당 평균 노출값으로 이미지들을 조절해서 합친다.

3. 구현시 참고사항
   - [명도(grayscale) 구하기](https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python)
   - [fix divide by zero](https://stackoverflow.com/questions/55653708/how-to-fix-runtimewarning-divide-by-zero-encountered-in-double-scalars/55653918)

---
### ✅ Level Four - 개발 후기
> 아무래도 Fujifilm의 기술이니까 자료가 많이 있을까 싶었는데, 의외로 정리된 자료들이 많아서 구현할 수 있었던 것 같다.
> 이번 기회에 openCV와 numpy의 기능을 많이 사용해본 것 같다. 아마 이후에 영상에 관련된 프로젝트에서도 많이 사용할 것 같으니 이번 기회에 잘 기억해 두어야할 것 같다.
> bright와 dark의 경우 처음에는 gpt의 설명을 믿고 '더 큰값과 더 작은 값만 구하면 되겠지~' 했다가 봉변을 당했다. (`역시 GPT는 참고만 해야한다...ㅠㅠ`)
> 역시나 가장 어려웠던 건 역시 average였던 것 같다. 처음에는 각 이미지의 노출값 평균을 가지고 픽셀마다 노출값 조정했다가 결과가 처참했다. 그래서 열심히 고민하고 찾아본 덕분에 원인이 '현재 기준에서 평균값에 도달하기 위해 얼만큼 조절해야하는가' 라는 점을 찾아내서 적용했고, 덕분에 만족스러운 average 기능이 구현될 수 있었다.
> 이번 기회를 기반으로 openCV로 영상처리를 할 때 좀 더 쉽게 다가갈 수 있을 것 같다.

---
### ✅ Level Five - 다음 단계는?
1. 해당 기능을 합리적이고 최저 성능의 IC 칩에 넣어서 동작시켜보기
2. 나만의 Toy Camera로 만들고 기능을 넣어보기
3. Fujifilm의 다른 재미있는 기능 구현해보기
