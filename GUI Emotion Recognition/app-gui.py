
from create_classifier import svm_classifer ,rf_classifer
from create_landmarks import start_capture
from create_landmarks import start_capture1

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import *

names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Emotion Recognizer")
        self.resizable(False, False)
        self.geometry("1400x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
            frame = self.frames[page_name]
            frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            f =  open("nameslist.txt", "a+")
            for i in names:
                    f.write(i+" ")
            self.destroy()


class StartPage(tk.Frame):

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            self.controller = controller
            render = PhotoImage(file='homepagepic.png')
            img = tk.Label(self, image=render, bd=0)
            img.image = render
            img.grid(row=0, column=5, rowspan=10, sticky="nsew")
            label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font,fg="#263942")
            label.grid(row=0, sticky="ew")
            button1 = tk.Button(self, text="   Start  ", fg="#ffffff", bg="#263942",command=lambda: self.controller.show_frame("PageOne"))
            button2 = tk.Button(self, text="   Check data  ", fg="#ffffff", bg="#263942",command=lambda: self.controller.show_frame("PageTwo"))
            button3 = tk.Button(self, text="Quit", fg="#263942", bg="#ffffff", command=self.on_closing)
            button1.grid(row=1, column=0, ipady=3, ipadx=7)
            button2.grid(row=2, column=0, ipady=3, ipadx=2)
            button3.grid(row=3, column=0, ipady=3, ipadx=32)

        def on_closing(self):
            if messagebox.askokcancel("Quit", "Are you sure?"):
                global names
                with open("nameslist.txt", "w") as f:
                    for i in names:
                        f.write(i + " ")
                self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="Enter the name", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942", command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)
    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller
        tk.Label(self, text="Select user", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, padx=10, pady=10)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"), bg="#ffffff", fg="#263942")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonload = tk.Button(self, text="Load", command=self.nextfoo, fg="#ffffff", bg="#263942")

        self.dropdown.grid(row=1, column=0, ipadx=8, padx=10, pady=10)
        self.buttoncanc.grid(row=3, ipadx=5, ipady=4, column=0, pady=10)
        self.buttonload.grid(row=1, ipadx=5, ipady=4, column=1, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        name = self.controller.active_name
        label = tk.Label
        path = "data/" + str(name) + "/" + str(name) + str(1) + ".jpg"
        load = Image.open(path)

        render = ImageTk.PhotoImage(load)
        img = label(self, image=render)
        img.image = render
        img.grid(row=0, column=0, rowspan=1, sticky="nsew")

    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text="Input image", font='Helvetica 12 bold', fg="#263942")
        self.numimglabel.grid(row=0, column=2, columnspan=3, sticky="ew", pady=10)
        self.capturebutton1 = tk.Button(self, text="Image netural", fg="#ffffff", bg="#263942", command=self.capimg1)
        self.capturebutton2 = tk.Button(self, text="Image emotion", fg="#ffffff", bg="#263942", command=self.capimg2)
        self.svmbutton = tk.Button(self, text="Suport Vector Machine Predict", fg="#ffffff", bg="#263942",command=self.SVMmodel)
        self.rfbutton = tk.Button(self, text="Random Forest Predict", fg="#ffffff", bg="#263942",command=self.RFmodel)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"), bg="#ffffff", fg="#263942")


        self.capturebutton1.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.capturebutton2.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)
        self.svmbutton.grid(row=2, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.rfbutton.grid(row=2, column=1, ipadx=5, ipady=4, padx=10, pady=20)
        self.buttoncanc.grid(row=2, ipadx=10, ipady=4, column=3, pady=20)


    def capimg1(self):

        messagebox.showinfo("INSTRUCTIONS", "Please sit still and look straight at the camera.")
        start_capture(self.controller.active_name)
        label = tk.Label
        name = self.controller.active_name
        path = "./data/" + str(name) +"/" + str(name) + ".jpg"
        load = Image.open(path)

        render = ImageTk.PhotoImage(load)
        img = label(self, image=render)
        img.image = render
        img.grid(row=0, column=0, rowspan=1, sticky="nsew")


    def capimg2(self):

        messagebox.showinfo("INSTRUCTIONS", "Please sit still and look straight at the camera.")
        start_capture1(self.controller.active_name)
        name = self.controller.active_name
        label = tk.Label
        path = "./data/" + str(name) + "/" + str(name) +str(1)  + ".jpg"
        load = Image.open(path)

        render = ImageTk.PhotoImage(load)
        img = label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=1, sticky="nsew")


    def SVMmodel(self):

        root = Tk()
        root.geometry("300x100")
        root.title("Support Vector Machine Predict")
        Output = Text(root, height=5,
                      width=25,
                      bg="light cyan")
        r = svm_classifer(self.controller.active_name)
        Output.insert("1.0", r + ", không bị Stress")
        Output.pack()


    def RFmodel(self):
        root = Tk()
        root.geometry("300x100")
        root.title("Random Forest Predict")
        Output = Text(root, height=5,
                      width=25,
                      bg="light cyan")
        r1 = rf_classifer(self.controller.active_name)

        Output.insert("1.0", r1 + ", không bị Stress")
        Output.pack()

from PIL import Image,ImageTk
app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
app.mainloop()

