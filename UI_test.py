import tkinter as tk
from tkinter import ttk

root = tk.Tk()

print(root.winfo_screenwidth()) #輸出螢幕寬度
print(root.winfo_screenheight()) #輸出螢幕高度
w=1280  #width
r=720  #height
x=100  #預設視窗位置(左上角x座標)
y=50  #預設視窗位置(左上角y座標)
root.geometry('%dx%d+%d+%d' % (w,r,x,y))
root.maxsize(root.winfo_screenwidth(), root.winfo_screenheight())  #設定視窗最大寬度和高度
root.minsize(800, 450)  #設定視窗最小寬度和高度
root.resizable(1, 1)  #設定視窗可調整大小
#root.iconify()  #設定視窗最小化
#root.state('zoomed')  #設定視窗最大化

root.title("My Application")  #設定視窗標題
root.configure(bg='#111111')  #設定視窗背景顏色
#root.iconbitmap("path/to/icon.ico")  #設定視窗圖示


L1=tk.Label(root,text='top fill x',bg='#555555',fg="#FFFFFF",
            font=("Inter",18,"bold")) #設定Label文字、背景顏色、前景顏色和字體
L2=tk.Label(root,text='left pad 10',bg='#555555',fg="#FFFFFF",
            font=("Inter",18,"bold"))
L3=tk.Label(root,text='right fill y',bg='#555555',fg="#FFFFFF",
            font=("Inter",18,"bold"))
L4=tk.Label(root,text='bottom ipad 20',bg='#555555',fg="#FFFFFF",
            font=("Inter",18,"bold"))
L5=tk.Label(root,text='center expand pad 1',bg='#555555',fg="#FFFFFF",
            font=("Inter",18,"bold"))

L1.pack(side="top", fill='x')  #設定Label位置和間距
L2.pack(side="left", padx=10, pady=10)
L3.pack(side="right", fill='y')
L4.pack(side="bottom", ipadx=20, ipady=20)
L5.pack(anchor="center", fill='both', padx=1, pady=1, expand=True) #anchor 參數有n, s, e, w, ne, nw, se, sw, center (方位)


'''
L1=tk.Label(root,text='0 0 cspan sticky EW',bg="#880000",fg="#FFFFFF",
            font=("Inter",18,"bold")) #設定Label文字、背景顏色、前景顏色和字體
L2=tk.Label(root,text='1 1 rspan sticky NS',bg="#00177C",fg="#FFFFFF",
            font=("Inter",18,"bold"))
L3=tk.Label(root,text='1 0 sticky EWNS',bg="#007A0A",fg="#FFFFFF",
            font=("Inter",18,"bold"))
L4=tk.Label(root,text='',bg="#680181",fg="#FFFFFF",
            font=("Inter",18,"bold"))

L1.grid(row=0, column=0, columnspan=2, sticky=tk.E+tk.W) #設定Label位置和間距
L2.grid(row=0, column=2, rowspan=2, sticky=tk.N+tk.S) #sticky參數有n, s, e, w (方位)
L3.grid(row=1, column=0, sticky=tk.E+tk.W+tk.N+tk.S) #columnspan和rowspan參數用於合併儲存格
L4.grid(row=1, column=1)
'''

'''
L1.place(x=0,y=0,height=100,width=200)#位置在(0,0)，高100寬200
L2.place(x=100,y=100,height=200,width=200)#位置在(100,100)，高200寬200
L3.place(x=250,y=250,height=150,width=200)#位置在(150,150)，高150寬200

L1.place(relx=0.1,rely=0.1,relheight=0.8,relwidth=0.8) #相對位置，relx和rely參數用於設定相對位置，relheight和relwidth參數用於設定相對高度和寬度

F1=tk.Frame(root, borderwidth=5, relief="ridge", width=90, height=50) #建立Frame容器
L1=tk.Label(F1, text="Welcome") #在Frame內建立Label
F2=tk.Frame(root, borderwidth=5, relief="ridge", width=90, height=50)
L2=tk.Label(F2, text="Welcome")


F1.pack()
F2.pack()
L1.place(x=10, y=10, bordermode="outside")#在父容器外
L2.place(x=10, y=10, bordermode="inside")#在父容器內
'''

'''
text=tk.Label(root, text='I am Label',
              height=7,width=25, #設定標籤高度為7寬度為25
              fg="#FF8000",bg="#02DF82", #更改前景與背景的顏色
              font=("Bauhaus 93",18,"bold","italic","underline"), #設定字型
              anchor='se') #設定標籤位置
text.pack()
'''

'''
R1 = tk.Button(root, text ="FLAT", relief="flat") #建立flat標籤
R2 = tk.Button(root, text ="RAISED", relief="raised") #建立raised標籤
R3 = tk.Button(root, text ="SUNKEN", relief="sunken") #建立sunken標籤
R4 = tk.Button(root, text ="GROOVE", relief="groove") #建立groove標籤
R5 = tk.Button(root, text ="RIDGE", relief="ridge") #建立ridge標籤

B1=tk.Label(root,bitmap="error")    #建立error位元圖
B2=tk.Label(root,bitmap="hourglass") #建立hourglass位元圖
B3=tk.Label(root,bitmap="info")     #建立info位元圖
B4=tk.Label(root,bitmap="questhead") #建立questhead位元圖
B5=tk.Label(root,bitmap="question") #建立question位元圖
B6=tk.Label(root,bitmap="warning") #建立warning位元圖
B7=tk.Label(root,bitmap="gray12") #建立gray12位元圖
B8=tk.Label(root,bitmap="gray25") #建立gray25位元圖
B9=tk.Label(root,bitmap="gray50") #建立gray50位元圖
B10=tk.Label(root,bitmap="gray75") #建立gray75位元圖
'''

'''
    text=tk.Label(root, text='I am Label',
                font="Times 25 bold",
                cursor='arrow') #設定滑鼠移到標籤上後的游標樣式
    text.pack()

    def clickHello():
    global count
    count=count + 1
    text.config(text="Click Hello " + str(count) + " times") #點擊按鈕後更改標籤文字
B=tk.Button(root, text="Hello", command=clickHello,font=("Bauhaus 93",20,"bold")) #建立按鈕並設定點擊事件
'''

root.mainloop()