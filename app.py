from tkinter import *
from tkinter.simpledialog import askstring
from tkinter import messagebox
from tkinter import filedialog as fd

from PIL import Image, ImageTk
import cv2 as cv
import algorithms as myDIP

class IMG_MenubarGUI(Menu):
    def __init__(self,root):
        self.root = root
        Menu.__init__(self,root)

        file = Menu(self, tearoff=1)    
        file.add_command(label="Load Image", command = self.open_file)  
        file.add_command(label="Clear Image", command = self.clear_img)  
        file.add_separator()  
        file.add_command(label="Exit", command=self.quit)  
        self.add_cascade(label="File", menu=file)  

        algorithms = Menu(self, tearoff=0)  
        
        subScale = Menu(algorithms)
        subScale.add_command(label="Nearest Neighbor Method", command=self.neighbor)
        subScale.add_command(label="X - Linear Method", command=lambda: self.linear("X"))
        subScale.add_command(label="Y - Linear Method", command=lambda: self.linear("Y"))
        subScale.add_command(label="Bilinear Interpolation Method", command=self.bilinear)
        algorithms.add_cascade(label="Scaling", menu = subScale)

        gScale = Menu(algorithms)
        gScale.add_command(label="8-bit", command=lambda: self.grey_scale(8))
        gScale.add_command(label="7-bit", command=lambda: self.grey_scale(7))
        gScale.add_command(label="6-bit", command=lambda: self.grey_scale(6))
        gScale.add_command(label="5-bit", command=lambda: self.grey_scale(5))
        gScale.add_command(label="4-bit", command=lambda: self.grey_scale(4))
        gScale.add_command(label="3-bit", command=lambda: self.grey_scale(3))
        gScale.add_command(label="2-bit", command=lambda: self.grey_scale(2))
        gScale.add_command(label="1-bit", command=lambda: self.grey_scale(1))
        algorithms.add_cascade(label="Grey Level Resolution", menu = gScale)  

        self.add_cascade(label="Algorithms", menu=algorithms)  

        help = Menu(self, tearoff=0)  
        help.add_command(label="About", command=self.about)  
        self.add_cascade(label="Help", menu=help)  
    
    def placeholder(self):
         messagebox.showinfo('Digital Image Processing', 'Sorry, the algorithm you are atempting to use has yet to be implemented.')
    def about(self):
        messagebox.showinfo('Digital Image Processing', 'CPP CS5550 Fall 2021')

    def neighbor(self):
        if self.root.dir_text.get() !="No image selected":
            im = cv.imread(str(self.root.dir_text.get()))
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            m = askstring('Scale', 'X:')
            n = askstring('Scale', 'Y:')
            self.root.disp_img(myDIP.nearest_Neighbor_Interpolation(im,(int(m),int(n))),self.root.imgLabel2,1,1)
            self.root.update_algo("Nearest Neighbor for " + str(m) + "*" + str(n))

    def linear(self,choice):
        if self.root.dir_text.get() !="No image selected":
            im = cv.imread(str(self.root.dir_text.get()))
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            m = askstring('Scale', 'X:')
            n = askstring('Scale', 'Y:')
            self.root.disp_img(myDIP.linear_Method(im,(int(m),int(n)),choice),self.root.imgLabel2,1,1)
            self.root.update_algo(choice + " - Linear Method for " + str(m) + "*" + str(n))

    def bilinear(self):
        if self.root.dir_text.get() !="No image selected":
            im = cv.imread(str(self.root.dir_text.get()))
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            m = askstring('Scale', 'X:')
            n = askstring('Scale', 'Y:')
            self.root.disp_img(myDIP.bilinear_Interpolation(im,(int(m),int(n))),self.root.imgLabel2,1,1)
            self.root.update_algo("Bilinear for " + str(m) + "*" + str(n))

    def grey_scale(self,level):
        if self.root.dir_text.get() !="No image selected":
            im = cv.imread(str(self.root.dir_text.get()))
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            self.root.disp_img(myDIP.grey_Level(im,level),self.root.imgLabel2,1,1)
            self.root.update_algo("Grey Scaling Level " + str(level))

    def open_file(self):
        filetypes = (('All files', '*.*'),)
        filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
        self.root.update_dir_label(filename)
        self.root.load_img()
    
    def clear_img(self):
        self.root.update_dir_label("No image selected")
        self.root.clear_img()
    
class IMG_MenuGUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.dir_text = StringVar()
        self.dir_text.set("No image selected")
        menubar = IMG_MenubarGUI(self)
        self.config(menu=menubar)
        
        label1 = Label(self,textvariable= self.dir_text)
        label1.grid(row = 0, column=0)

        self.imgLabel1 = Label(self)
        self.imgLabel2 = Label(self)

        self.algro_text = StringVar()
        self.algro_text.set("Algorithm: None")
        self.algrLabel = Label(self,textvariable=self.algro_text)

    def update_dir_label(self, menubar):
        self.dir_text.set(menubar)

    def clear_img(self):
        self.imgLabel1.grid_forget()
        self.imgLabel2.grid_forget()
        self.algrLabel.grid_forget()
       
    def disp_img(self,img,imgPlot,m,n):
         # convert the images to PIL format...
        image = Image.fromarray(img)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        imgPlot = Label(image=image)
        imgPlot.image = image
        imgPlot.grid(row=m,column=n, padx=10, pady=10)

    def load_img(self):
        im = cv.imread(str(self.dir_text.get()))
        im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        self.disp_img(im,self.imgLabel1,1,0)
        self.algrLabel.grid(row=2,column=0)
    
    def update_algo(self,msg):
        self.algro_text.set("Algorithm: " + msg)
    

if __name__ == "__main__":
    root = IMG_MenuGUI()
    root.title("Digital Image Processing GUI")
    root.iconbitmap('./favicon.ico')
    root.geometry('1280x720')

    root.mainloop()