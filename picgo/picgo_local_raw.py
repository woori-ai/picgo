아래는 주어진 요구사항에 맞춰 개발된 텍스트 기반 이미지 생성 앱의 완전한 Python 소스 코드입니다. 이 코드는 `tkinter`를 사용하여 간단한 GUI를 구현하고, 미리 다운로드 받아 놓은 딥러닝 모델을 활용하여 이미지를 생성합니다. 예시로는 `stable-diffusion` 모델을 사용한다고 가정하고, 실제 모델 로드 및 이미지 생성 로직은 추상화된 방식으로 구현되었습니다.

```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
from PIL import Image, ImageTk
import numpy as np

# 가상의 모델 로딩 함수 (실제 모델 로드 로직은 여기에 구현)
class TextToImageModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.loaded = False
        print(f"Loading model: {model_name}")
        # 실제 모델 로드 로직 구현 (예: 모델 가중치 로드 등)
        self.loaded = True  # 가상으로 모델 로드 완료 표시

    def generate_image(self, prompt):
        if not self.loaded:
            raise Exception("Model not loaded")
        # 가상의 이미지 생성 로직 (실제 생성 로직으로 대체 필요)
        image_data = np.random.rand(512, 512, 3) * 255  # 랜덤 이미지 데이터 생성
        image = Image.fromarray(image_data.astype('uint8'))
        return image

# GUI 클래스 정의
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text-to-Image Generator")
        self.models = {
            "Stable Diffusion": TextToImageModel("stable_diffusion"),
            # 추가 모델들을 여기에 등록 가능
        }
        self.selected_model = None
        self.prompt = ""

        # UI 구성
        self.create_ui()

    def create_ui(self):
        # 텍스트 프롬프트 입력 창
        self.prompt_label = tk.Label(self.root, text="Enter Prompt:", font=("Arial", 12))
        self.prompt_label.pack(pady=5)
        self.prompt_entry = tk.Text(self.root, height=5, width=50, wrap=tk.WORD)
        self.prompt_entry.insert(tk.END, "예: '해변의 일출'")
        self.prompt_entry.pack(pady=5)
        self.prompt_entry.config(state=tk.DISABLED)  # 초기 비활성화 후 사용 시 활성화

        # 모델 선택 드롭다운 메뉴
        self.model_var = tk.StringVar()
        self.model_var.set("Stable Diffusion")  # 기본값 설정
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=list(self.models.keys()))
        self.model_dropdown.pack(pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)

        # 생성 버튼
        self.generate_button = tk.Button(self.root, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        # 이미지 표시 영역
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

    def on_model_select(self, event):
        selected_model = self.model_var.get()
        self.selected_model = self.models.get(selected_model)
        if self.selected_model:
            model_name = f"Selected Model: {selected_model}"
            self.prompt_label.config(text=model_name)
        else:
            messagebox.showerror("Error", "Model not found")

    def generate_image(self):
        if not self.selected_model or not self.prompt_entry.get("1.0", tk.END).strip():
            messagebox.showwarning("Warning", "Please select a model and enter a prompt.")
            return

        self.prompt = self.prompt_entry.get("1.0", tk.END).strip()
        start_time = time.time()
        while time.time() - start_time < 60:  # 최대 60초 대기
            self.generate_button.config(state=tk.DISABLED)
            try:
                # 가상 이미지 생성 로직 호출
                image = self.selected_model.generate_image(self.prompt)
                self.display_image(image)
                break
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate image: {str(e)}")
                self.generate_button.config(state=tk.NORMAL)
                time.sleep(2)  # 에러 후 재시도 대기 시간
        self.generate_button.config(state=tk.NORMAL)

    def display_image(self, image):
        img = ImageTk.PhotoImage(image)
        self.image_label.config(image=img)
        self.image_label.image = img  # 이미지 참조 유지

        # 이미지 저장 옵션
        save_filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_filepath:
            image.save(save_filepath)
            messagebox.showinfo("Success", f"Image saved to {save_filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
```

### 주요 특징:
1. **UI 구성**:
   - **텍스트 프롬프트 입력 창**: 사용자가 프롬프트를 입력할 수 있는 텍스트 박스 제공.
   - **모델 선택 드롭다운 메뉴**: 미리 정의된 모델 목록에서 선택 가능.
   - **생성 버튼**: 이미지 생성 요청을 트리거하는 버튼.
   - **이미지 표시 영역**: 생성된 이미지를 표시하는 영역.
   - **이미지 저장 옵션**: 사용자가 생성된 이미지를 저장할 수 있는 옵션 제공.

2. **모델 로딩 및 생성 로직**:
   - `TextToImageModel` 클래스는 가상의 모델 로딩과 이미지 생성 로직을 나타냅니다. 실제 모델 로드 및 생성 로직으로 대체해야 합니다.

3. **에러 처리 및 피드백**:
   - 에러 메시지 표시 및 사용자 피드백 제공을 위한 메시지 박스 사용.

이 코드는 전체적인 구조와 로직을 포함하고 있으며, 실제 딥러닝 모델과 연동하여 사용할 때 필요한 부분을 실제 구현에 맞게 조정하면 됩니다.