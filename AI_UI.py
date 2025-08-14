import tkinter as tk
from tkinter import filedialog

def train():
    print("訓練按鈕被點擊")

def detect():
    print("偵測按鈕被點擊")

root = tk.Tk()

w=800  #width
r=600  #height
x=550  #預設視窗位置(左上角x座標)
y=180  #預設視窗位置(左上角y座標)
root.geometry('%dx%d+%d+%d' % (w,r,x,y))
root.resizable(0, 0)  #設定視窗可調整大小

root.title("簡易網路封包異常偵測系統")  #設定視窗標題
root.configure(bg='#111111')  #設定視窗背景顏色

title_label = tk.Label(root, text="簡易網路封包異常偵測系統", bg="#111111", fg="#A5A5A5", font=("Inter", 24, "bold"))
title_label.pack(side="top", pady=10)

subtitle_label = tk.Label(root, text="請選擇資料型態後選擇功能", bg="#111111", fg="#A5A5A5", font=("Inter", 18))
subtitle_label.pack(side="top", pady=5)

train_btn = tk.Button(root, text="訓練", bg="#414141", fg="#a3a3a3", font=("Inter", 18), width=22, height=10, command=train)
train_btn.pack(side="left", anchor="nw", padx=40, fill='none')

detect_btn = tk.Button(root, text="偵測", bg="#414141", fg="#a3a3a3", font=("Inter", 18), width=22, height=10, command=detect)
detect_btn.pack(side="right", anchor="ne", padx=40, fill='none')

# 創建Frame
file_frame = tk.Frame(root, bg="#111111")
file_frame.pack(side="bottom", fill='x')

# Radio選項
data_type = tk.StringVar(value="option1")
radio1 = tk.Radiobutton(file_frame, text="選項1", variable=data_type, value="option1", bg="#111111", fg="#A5A5A5", font=("Inter", 14), selectcolor="#222222")
radio2 = tk.Radiobutton(file_frame, text="選項2", variable=data_type, value="option2", bg="#111111", fg="#A5A5A5", font=("Inter", 14), selectcolor="#222222")
radio1.pack(side="left", padx=10)
radio2.pack(side="left", padx=10)

# 檔案路徑顯示
file_path_var = tk.StringVar()

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_path_var.set(file_path)  # 文字框會自動顯示選擇的檔案路徑

# 瀏覽檔案按鈕
browse_btn = tk.Button(file_frame, text="瀏覽檔案", command=browse_file, bg="#414141", fg="#a3a3a3", font=("Inter", 12))
browse_btn.pack(side="top", pady=(20, 5))

# 顯示檔案路徑的文字框
file_entry = tk.Entry(file_frame, textvariable=file_path_var, width=60, font=("Inter", 12))
file_entry.pack(side="top", pady=5)

root.mainloop()