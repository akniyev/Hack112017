from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import glob


def cbc(obj):
    global ind
    if ind > -1:
        img = files[ind]
        global T
        y = T.get("1.0", END)
        targetfile = img.replace(".jpg", "") + "_y.txt"
        with open(targetfile, "w") as f:
            f.write(y)
        T.delete("1.0", END)

    if ind >= len(files)-1:
        messagebox.showinfo("Done", "Done")
        return
    ind = ind + 1
    img = files[ind]
    global L
    image = Image.open(img)
    photo = ImageTk.PhotoImage(image)
    L.configure(image=photo)
    L.image = photo
    L.pack()



target_folder = 'h:/Dropbox/INFO_BASE_EXT/Hack112017/**/*.jpg'
files = []
ind = -1
for f in glob.glob(target_folder, recursive=True):
    files.append(f)

root = Tk()
root.bind("<Return>", cbc)

photo = ImageTk.PhotoImage(Image.open('labels.png'))
Label(root, image=photo).pack()


T = Text(root, height=2, width=30)
T.pack()

B = Button(root, text='Next', command=cbc)
B.pack()

L = Label(root)
L.pack()

root.mainloop()
