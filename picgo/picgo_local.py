import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
from PIL import Image, ImageTk

# Dependency Check
try:
    import torch
    from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline
    DEPENDENCIES_INSTALLED = True
except ImportError as e:
    DEPENDENCIES_INSTALLED = False
    MISSING_LIB = str(e).split("'")[1] if "'" in str(e) else str(e)

# ---------------------------------------------------------
# 모델 로딩 및 생성 클래스 (Diffusers 기반)
# ---------------------------------------------------------
class TextToImageModel:
    def __init__(self):
        self.pipe = None
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_error = ""

    def load_model(self, model_path_or_id):
        print(f"Loading model from: {model_path_or_id} on {self.device}...")
        self.last_error = ""  # Clear previous errors
        try:
            # 로컬 파일(.safetensors/.ckpt)
            if os.path.isfile(model_path_or_id):
                # 1. SDXL 시도 (우선순위 높임: 1.5 파이프라인으로 잘못 로드되는 것 방지)
                try:
                    print("Attempting to load as SDXL Checkpoint...")
                    self.pipe = StableDiffusionXLPipeline.from_single_file(
                        model_path_or_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        use_safetensors=True
                    )
                except Exception as e_sdxl:
                    print(f"SDXL load failed: {e_sdxl}")
                    error_str = str(e_sdxl)
                    
                    # SDXL 컴포넌트 누락 시 복구 로직
                    if any(x in error_str for x in ["CLIPTextModel", "UNet2DConditionModel", "AutoencoderKL", "tokenize", "scheduler"]):
                        print("Missing components detected. Downloading standard SDXL components from HuggingFace...")
                        try:
                            from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
                            from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
                            
                            print("Loading standard SDXL components...")
                            text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
                            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
                            tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
                            tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
                            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
                            vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
                            scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

                            self.pipe = StableDiffusionXLPipeline.from_single_file(
                                model_path_or_id,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                unet=unet,
                                vae=vae,
                                scheduler=scheduler,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                use_safetensors=True
                            )
                        except Exception as e_fallback:
                            print(f"SDXL fallback failed: {e_fallback}")
                            # 실패 시 무시하고 아래 SD1.5 시도로 넘어감
                            pass
                    else:
                         print("Not a component error. Trying SD1.5...")
                    
                    # SDXL 실패 시 (파이프라인이 생성되지 않았으면) SD1.5 시도
                    if not self.pipe:
                        try:
                            print("Attempting to load as SD1.5/2.1 Checkpoint...")
                            self.pipe = StableDiffusionPipeline.from_single_file(
                                model_path_or_id, 
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                use_safetensors=True
                            )
                        except Exception as e_sd:
                             raise Exception(f"Failed to load as both SDXL and SD1.5.\nSDXL error: {e_sdxl}\nSD1.5 error: {e_sd}")

            # HuggingFace Model ID
            else:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_path_or_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            
            self.pipe.to(self.device)
            # Mac(MPS) 최적화
            if self.device == "mps":
                self.pipe.enable_attention_slicing()
            
            self.model_path = model_path_or_id
            print("Model loaded successfully.")
            return True
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            print(msg)
            self.last_error = str(e)
            return False

    def set_device(self, device_name):
        if device_name == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_name
        
        print(f"Device set to: {self.device}")
        if self.pipe:
            self.pipe.to(self.device)

    def generate_image(self, prompt, negative_prompt=""):
        if not self.pipe:
            raise Exception("Model is not loaded.")
        
        # 이미지 생성
        image = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        return image

# ---------------------------------------------------------
# GUI 클래스
# ---------------------------------------------------------
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PicGo Local - AI Image Generator")
        self.root.geometry("600x850") # Height increased
        
        self.model_engine = TextToImageModel()

        if not DEPENDENCIES_INSTALLED:
            messagebox.showerror("Missing Libraries", f"핵심 라이브러리({MISSING_LIB})가 설치되지 않았습니다.\n\n터미널에서 아래 명령어를 실행하세요:\npip install torch diffusers transformers accelerate")
        
        self.create_ui()

    def create_ui(self):
        # Top Frame Container for Model Settings and Help
        frame_top = tk.Frame(self.root)
        frame_top.pack(fill="x", padx=10, pady=5)

        # 1. 모델 선택 영역 (2/3 Width)
        frame_model = tk.LabelFrame(frame_top, text="Model Settings", padx=10, pady=10)
        frame_model.pack(side="left", fill="both", expand=True, padx=(0, 5)) 
        
        self.lbl_model = tk.Label(frame_model, text="No model loaded", fg="red")
        self.lbl_model.pack(side="left", expand=True)
        
        btn_load = tk.Button(frame_model, text="Load Model...", command=self.load_model_dialog)
        btn_load.pack(side="right")

        # 1.5 Help Button Area (1/3 approx visual weight)
        frame_help = tk.LabelFrame(frame_top, text="Help & Info", padx=10, pady=10)
        frame_help.pack(side="right", fill="both", padx=(5, 0))
        
        btn_help = tk.Button(frame_help, text="Help / Download Models", command=self.open_help, bg="lightyellow")
        btn_help.pack(fill="both", expand=True)

        # 1.6 디바이스 선택
        self.create_device_selection_ui()

        # 2. 프롬프트 입력 영역
        frame_prompt = tk.LabelFrame(self.root, text="Prompts", padx=10, pady=10)
        frame_prompt.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_prompt, text="Positive Prompt:").pack(anchor="w")
        self.txt_prompt = tk.Text(frame_prompt, height=3, width=50)
        self.txt_prompt.pack(fill="x", pady=2)
        
        tk.Label(frame_prompt, text="Negative Prompt:").pack(anchor="w")
        self.txt_neg_prompt = tk.Text(frame_prompt, height=2, width=50)
        self.txt_neg_prompt.pack(fill="x", pady=2)
        self.txt_neg_prompt.insert("1.0", "low quality, bad anatomy, blurry")

        # 3. 실행 버튼
        self.btn_generate = tk.Button(self.root, text="Generate Image", command=self.start_generation, bg="lightblue", font=("Arial", 12, "bold"))
        self.btn_generate.pack(fill="x", padx=10, pady=10)

        # 4. 이미지 표시 영역
        self.lbl_image = tk.Label(self.root, text="Generated image will appear here", bg="#f0f0f0", height=20)
        self.lbl_image.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 5. 저장 버튼
        self.btn_save = tk.Button(self.root, text="Save Image", command=self.save_image, state="disabled")
        self.btn_save.pack(pady=10)

        self.current_image = None

    def create_device_selection_ui(self):
        """디바이스 선택(토글) UI 생성 함수"""
        frame_device = tk.LabelFrame(self.root, text="Device Settings (Performance)", padx=10, pady=10)
        frame_device.pack(fill="x", padx=10, pady=5)

        self.device_var = tk.StringVar(value="auto")
        
        # 토글 버튼 (Radiobutton)
        rb_auto = tk.Radiobutton(frame_device, text="Auto (Recommended)", variable=self.device_var, value="auto", command=self.on_device_change)
        rb_cpu = tk.Radiobutton(frame_device, text="CPU (Slow)", variable=self.device_var, value="cpu", command=self.on_device_change)
        rb_cuda = tk.Radiobutton(frame_device, text="GPU (CUDA/NVIDIA)", variable=self.device_var, value="cuda", command=self.on_device_change)
        
        rb_auto.pack(side="left", padx=10)
        rb_cpu.pack(side="left", padx=10)
        rb_cuda.pack(side="left", padx=10)

    def open_help(self):
        help_win = tk.Toplevel(self.root)
        help_win.title("PicGo Help")
        help_win.geometry("500x400")
        
        lbl_title = tk.Label(help_win, text="PicGo Local Guide", font=("Arial", 14, "bold"))
        lbl_title.pack(pady=10)
        
        info_text = (
            "1. 모델 다운로드 방법:\n"
            "   - Civitai (https://civitai.com) 또는 HuggingFace에서\n"
            "   - 'SDXL Base' 또는 'SD 1.5' 기반의 Checkpoint(.safetensors)를 다운로드하세요.\n"
            "   - 다운로드한 파일을 'picgo/model' 폴더에 넣으세요.\n\n"
            "2. 사용 방법:\n"
            "   - 'Load Model...'로 모델을 선택합니다.\n"
            "   - 프롬프트를 입력하고 'Generate Image'를 누르세요.\n\n"
            "3. 성능:\n"
            "   - NVIDIA 그래픽카드가 있다면 'GPU' 모드를 권장합니다.\n"
        )
        
        lbl_info = tk.Label(help_win, text=info_text, justify="left", padx=20)
        lbl_info.pack(anchor="w")
        
        import webbrowser
        def open_link(url):
            webbrowser.open(url)
            
        btn_link = tk.Button(help_win, text="Go to Civitai (Models)", command=lambda: open_link("https://civitai.com"), fg="blue", cursor="hand2")
        btn_link.pack(pady=5)
        
        btn_close = tk.Button(help_win, text="Close", command=help_win.destroy)
        btn_close.pack(pady=20)

    def on_device_change(self):
        """토글 변경 시 하드웨어 체크 및 적용"""
        selected_device = self.device_var.get()
        
        # 하드웨어 체크
        if not self.check_hardware_availability(selected_device):
            # 체크 실패 시 CPU로 강제 변경 (또는 Auto)
            self.device_var.set("cpu")
            selected_device = "cpu"
        
        self.model_engine.set_device(selected_device)

    def check_hardware_availability(self, device_mode):
        """선택한 장치가 실제로 사용 가능한지 확인"""
        if device_mode == "cuda":
            if not torch.cuda.is_available():
                messagebox.showwarning(
                    "GPU Not Found", 
                    "NVIDIA GPU(CUDA)가 감지되지 않았습니다.\n\n"
                    "CPU 모드로 자동 전환됩니다.\n"
                    "GPU를 사용하려면 CUDA 지원 PyTorch를 다시 설치해야 합니다."
                )
                return False
        return True

    def load_model_dialog(self):
        if not DEPENDENCIES_INSTALLED:
            messagebox.showerror("Error", "Required libraries (torch, diffusers) are missing.")
            return

        # PyInstaller 빌드 시 경로 처리
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(__file__)

        initial_dir = os.path.join(base_path, "model")
        if not os.path.exists(initial_dir):
            os.makedirs(initial_dir, exist_ok=True)

        file_path = filedialog.askopenfilename(
            title="Select Model File (.safetensors, .ckpt)",
            initialdir=initial_dir,
            filetypes=[("Model checkpotins", "*.safetensors *.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.lbl_model.config(text="Loading...", fg="orange")
            self.root.update()
            
            # 스레드에서 로딩
            threading.Thread(target=self._load_model_thread, args=(file_path,), daemon=True).start()

    def _load_model_thread(self, path):
        success = self.model_engine.load_model(path)
        if success:
            self.root.after(0, lambda: self.lbl_model.config(text=f"Loaded: {os.path.basename(path)}", fg="green"))
        else:
            self.root.after(0, lambda: self.lbl_model.config(text="Load Failed", fg="red"))
            # 에러 메시지를 팝업에 표시
            error_msg = self.model_engine.last_error
            self.root.after(0, lambda: messagebox.showerror("Load Failed", f"모델 로딩 실패:\n{error_msg}"))

    def start_generation(self):
        if not self.model_engine.pipe:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        prompt = self.txt_prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt.")
            return
            
        # Main Thread에서 UI 값 읽기
        neg_prompt = self.txt_neg_prompt.get("1.0", "end").strip()
        
        self.btn_generate.config(state="disabled", text="Generating...", bg="orange")
        
        # 스레드에서 생성 (데이터 전달)
        threading.Thread(target=self._generate_thread, args=(prompt, neg_prompt), daemon=True).start()

    def _generate_thread(self, prompt, neg_prompt):
        try:
            image = self.model_engine.generate_image(prompt, neg_prompt)
            self.root.after(0, lambda: self.show_image(image))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            try:
                with open("picgo_error.log", "w", encoding="utf-8") as f:
                    f.write(tb)
            except:
                pass
            
            self.root.after(0, lambda: messagebox.showerror("Generation Error", f"{str(e)}\n\n(See picgo_error.log)"))
        finally:
            self.root.after(0, lambda: self.btn_generate.config(state="normal", text="Generate Image", bg="lightblue"))

    def show_image(self, image):
        self.current_image = image
        # 리사이즈해서 표시
        display_size = (512, 512)
        img_resized = image.resize(display_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_resized)
        
        self.lbl_image.config(image=tk_img, text="")
        self.lbl_image.image = tk_img # 참조 유지
        self.btn_save.config(state="normal")

    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.current_image.save(file_path)
                messagebox.showinfo("Saved", f"Image saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
