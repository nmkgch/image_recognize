from tkinter import filedialog

def select_image_file():
    typ = [('テキストファイル','*.jpg')] 
    dir = 'C:\\pg'
    fle = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 

    return fle