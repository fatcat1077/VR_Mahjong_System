import tkinter as tk
import pyautogui

def update_position():
    x, y = pyautogui.position()
    label.config(text=f"目前滑鼠位置：x={x}, y={y}")
    # 每 50 毫秒更新一次
    root.after(50, update_position)

root = tk.Tk()
root.title("滑鼠座標顯示")

label = tk.Label(root, text="目前滑鼠位置：x=0, y=0", font=("Arial", 14))
label.pack(padx=20, pady=20)

update_position()
root.mainloop()
