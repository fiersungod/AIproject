import tkinter as tk
import os

root = tk.Tk()

# ------------------------
# 視窗設定
# ------------------------
w, r, x, y = 800, 600, 550, 180
root.geometry('%dx%d+%d+%d' % (w, r, x, y))
root.resizable(0, 0)
root.title("簡易網路封包異常偵測系統")
root.configure(bg='#111111')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # UI所在資料夾
DATA_FOLDER = os.path.join(BASE_DIR, "local_data_set")
MODEL_FOLDER = os.path.join(BASE_DIR, "save_model")

# ------------------------
# 切換頁面功能
# ------------------------
def show_frame(frame):
    frame.tkraise()  # 把 frame 提到最上層
    # 如果切換到訓練頁面，自動載入檔案
    if frame == train_frame:
        load_files()
    elif frame == detect_frame:
        load_detect_models()

# ------------------------
# 建立容器 (放所有頁面)
# ------------------------
container = tk.Frame(root, bg="#111111")
container.pack(fill="both", expand=True)

# ------------------------
# 首頁 (main_frame)
# ------------------------
main_frame = tk.Frame(container, bg="#111111")
main_frame.place(relwidth=1, relheight=1)

title_label = tk.Label(main_frame, text="簡易網路封包異常偵測系統",
                       bg="#111111", fg="#A5A5A5", font=("Inter", 24, "bold"))
title_label.pack(pady=10)

subtitle_label = tk.Label(main_frame, text="請選擇資料型態後選擇功能",
                          bg="#111111", fg="#A5A5A5", font=("Inter", 18))
subtitle_label.pack(pady=5)

train_btn = tk.Button(main_frame, text="訓練", bg="#414141", fg="#a3a3a3",
                      font=("Inter", 18), width=22, height=10,
                      command=lambda: show_frame(train_frame))
train_btn.pack(side="left", padx=40)

detect_btn = tk.Button(main_frame, text="偵測", bg="#414141", fg="#a3a3a3",
                       font=("Inter", 18), width=22, height=10,
                       command=lambda: show_frame(detect_frame))
detect_btn.pack(side="right", padx=40)

# ------------------------
# 訓練頁面 (train_frame)
# ------------------------
train_frame = tk.Frame(container, bg="#111111")
train_frame.place(relwidth=1, relheight=1)

tk.Label(train_frame, text="訓練頁面", bg="#111111", fg="white", font=("Inter", 20, "bold")).pack(pady=10)

# 上方分成左右兩個區域
content_frame = tk.Frame(train_frame, bg="#111111")
content_frame.pack(fill="both", expand=True, padx=20, pady=10)

# 存放變數
file_vars = {}
model_var = tk.StringVar(value="")  # Radiobutton 單選變數

# ------------------------
# 左邊：訓練資料 (多選)
# ------------------------
left_frame = tk.Frame(content_frame, bg="#111111")
left_frame.pack(side="left", fill="both", expand=True, padx=10)

tk.Label(left_frame, text="選擇訓練資料", bg="#111111", fg="#A5A5A5", font=("Inter", 16, "bold")).pack(pady=5)

file_canvas = tk.Canvas(left_frame, bg="#111111", highlightthickness=0)
file_scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=file_canvas.yview)
file_scrollable = tk.Frame(file_canvas, bg="#111111")

file_scrollable.bind(
    "<Configure>",
    lambda e: file_canvas.configure(scrollregion=file_canvas.bbox("all"))
)

file_canvas.create_window((0, 0), window=file_scrollable, anchor="nw")
file_canvas.configure(yscrollcommand=file_scrollbar.set)

file_canvas.pack(side="left", fill="both", expand=True)
file_scrollbar.pack(side="right", fill="y")

# ------------------------
# 右邊：模型選擇 (單選)
# ------------------------
right_frame = tk.Frame(content_frame, bg="#111111")
right_frame.pack(side="left", fill="both", expand=True, padx=10)

tk.Label(right_frame, text="選擇模型", bg="#111111", fg="#A5A5A5", font=("Inter", 16, "bold")).pack(pady=5)

model_canvas = tk.Canvas(right_frame, bg="#111111", highlightthickness=0)
model_scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=model_canvas.yview)
model_scrollable = tk.Frame(model_canvas, bg="#111111")

model_scrollable.bind(
    "<Configure>",
    lambda e: model_canvas.configure(scrollregion=model_canvas.bbox("all"))
)

model_canvas.create_window((0, 0), window=model_scrollable, anchor="nw")
model_canvas.configure(yscrollcommand=model_scrollbar.set)

model_canvas.pack(side="left", fill="both", expand=True)
model_scrollbar.pack(side="right", fill="y")

# ------------------------
# 載入檔案
# ------------------------
def load_files():
    # 清空舊的元件
    for widget in file_scrollable.winfo_children():
        widget.destroy()
    for widget in model_scrollable.winfo_children():
        widget.destroy()
    file_vars.clear()
    model_var.set("")

    # 訓練資料 (Checkbutton，多選)
    try:
        files = os.listdir(DATA_FOLDER)
        for f in files:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(file_scrollable, text=f, variable=var,
                                 bg="#111111", fg="#A5A5A5", selectcolor="#222222",
                                 anchor="w", font=("Inter", 12))
            chk.pack(fill="x", padx=5, pady=2, anchor="w")
            file_vars[f] = var
    except Exception as e:
        tk.Label(file_scrollable, text=f"讀取資料夾錯誤: {e}", bg="#111111", fg="red").pack()

    # 模型 (Radiobutton，單選)
    try:
        models = os.listdir(MODEL_FOLDER)
        for m in models:
            rdo = tk.Radiobutton(model_scrollable, text=m, variable=model_var, value=m,
                                 bg="#111111", fg="#A5A5A5", selectcolor="#222222",
                                 anchor="w", font=("Inter", 12))
            rdo.pack(fill="x", padx=5, pady=2, anchor="w")
    except Exception as e:
        tk.Label(model_scrollable, text=f"讀取資料夾錯誤: {e}", bg="#111111", fg="red").pack()

# ------------------------
# 下方按鈕
# ------------------------
def start_training():
    selected_files = [f for f, v in file_vars.items() if v.get()]
    selected_model = model_var.get()
    print("選擇的訓練資料:", selected_files)
    print("選擇的模型:", selected_model)

tk.Button(train_frame, text="開始訓練", command=start_training,
          bg="#414141", fg="#a3a3a3", font=("Inter", 14), width=20, height=2).pack(pady=20)

tk.Button(train_frame, text="返回首頁", command=lambda: show_frame(main_frame)).pack(pady=10)

# ------------------------
# 偵測頁面 (detect_frame)
# ------------------------
detect_frame = tk.Frame(container, bg="#111111")
detect_frame.place(relwidth=1, relheight=1)

tk.Label(detect_frame, text="偵測頁面", bg="#111111", fg="white", font=("Inter", 20, "bold")).pack(pady=10)

# 模型選擇 (單選)

detect_model_var = tk.StringVar(value="")
detect_model_canvas = tk.Canvas(detect_frame, bg="#111111", highlightthickness=0)
detect_model_scrollbar = tk.Scrollbar(detect_frame, orient="vertical", command=detect_model_canvas.yview)
detect_model_scrollable = tk.Frame(detect_model_canvas, bg="#111111")
detect_model_scrollable.bind("<Configure>", lambda e: detect_model_canvas.configure(scrollregion=detect_model_canvas.bbox("all")))
detect_model_canvas.create_window((0,0), window=detect_model_scrollable, anchor="nw")
detect_model_canvas.configure(yscrollcommand=detect_model_scrollbar.set)
detect_model_canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
detect_model_scrollbar.pack(side="right", fill="y", pady=10)

def load_detect_models():
    for widget in detect_model_scrollable.winfo_children():
        widget.destroy()
    detect_model_var.set("")
    try:
        models = os.listdir(MODEL_FOLDER)
        for m in models:
            rdo = tk.Radiobutton(detect_model_scrollable, text=m, variable=detect_model_var, value=m,
                                 bg="#111111", fg="#A5A5A5", selectcolor="#222222",
                                 anchor="w", font=("Inter", 12))
            rdo.pack(fill="x", padx=5, pady=2, anchor="w")
    except Exception as e:
        tk.Label(detect_model_scrollable, text=f"讀取資料夾錯誤: {e}", bg="#111111", fg="red").pack()

def real_time_detect():
    selected_model = detect_model_var.get()
    if selected_model:
        print("即時偵測，使用模型:", selected_model)
    else:
        print("請先選擇模型！")

tk.Button(detect_frame, text="即時偵測", command=real_time_detect,
          bg="#414141", fg="#a3a3a3", font=("Inter", 14), width=20, height=2).pack(pady=10)
tk.Button(detect_frame, text="返回首頁", command=lambda: show_frame(main_frame)).pack(pady=10)

# ------------------------
# 預設顯示首頁
# ------------------------
for frame in (main_frame, train_frame, detect_frame):
    frame.place(relwidth=1, relheight=1)

show_frame(main_frame)

root.mainloop()
