from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as fd

class IMG_MenubarGUI(Menu):
    def __init__(self,root):
        self.root = root
        Menu.__init__(self,root)

        file = Menu(self, tearoff=1)  
        file.add_command(label="New")  
        file.add_command(label="Open", command = self.open_file)  
        file.add_command(label="Save")  
        file.add_command(label="Save as")    
        file.add_separator()  
        file.add_command(label="Exit", command=self.quit)  
        self.add_cascade(label="File", menu=file)  

        edit = Menu(self, tearoff=0)  
        edit.add_command(label="Undo")  
        edit.add_separator()     
        edit.add_command(label="Cut")  
        edit.add_command(label="Copy")  
        edit.add_command(label="Paste")  
        self.add_cascade(label="Edit", menu=edit)  

        help = Menu(self, tearoff=0)  
        help.add_command(label="About", command=self.about)  
        self.add_cascade(label="Help", menu=help)  
    
    def about(self):
        messagebox.showinfo('Digital Image Processing', 'CPP CS5550 Fall 2021')

    def open_file(self):
        filetypes = (('All files', '*.*'),)
        filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
        self.root.update_dir_label(filename)
    
class IMG_MenuGUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.dir_text = StringVar()
        self.dir_text.set("No image selected")
        menubar = IMG_MenubarGUI(self)
        self.config(menu=menubar)
        
        label1 = Label(self,textvariable= self.dir_text)
        label1.grid(row = 0, column=0)
    
    def update_dir_label(self, menubar):
        self.dir_text.set(menubar)
        

if __name__ == "__main__":
    root = IMG_MenuGUI()
    root.title("Digital Image Processing GUI")
    root.iconbitmap('./favicon.ico')
    root.geometry('1280x720')

    root.mainloop()