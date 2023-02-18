from tkinter import *
from tkinter import messagebox
import os

root=Tk()
root.title('Login')
root.geometry('400x500+300+200')
root.configure(bg = '#FFF')
root.resizable(False,False)

def signin():
    username = user.get()
    password = code.get()

    if username =='admin' and password =='1234' :
        os.system('sudo -u bhuvan python3.8 /home/bhuvan/Desktop/Code/email/GUI/GUI.py')
frame = Frame(root,width = 350,height = 350, bg = 'white')
frame.place(x=30,y=70)

heading = Label(frame,text = 'Sign in', fg = '#57a1f8',bg='white', font =('Microsoft YaHei UI Light',23,'bold'))
heading.place(x=100,y=5)


def on_enter(e):
    user.delete(0,'end')

def on_leave(e):
    name = user.get()
    if name == "":
        user.insert(0,'Username')

#Username Tab
user = Entry(frame,width=25,fg='black', border=0, bg='white',font =('Microsoft YaHei UI Light',11))
user.place(x=45, y=80)
user.insert(0,'Username')
user.bind('<FocusIn>', on_enter)
user.bind('<FocusOut>', on_leave)

Frame(frame,width=240,height=2,bg='black').place(x=45,y=105)

#Password Tab
def on_enter(e):
    code.delete(0,'end')

def on_leave(e):
    name = code.get()
    if name == "":
        user.insert(0,'Password')

code = Entry(frame,width=25,fg='black', border=0, bg='white',font =('Microsoft YaHei UI Light',11))
code.place(x=45, y=160)
code.insert(0,'Password')
code.bind('<FocusIn>', on_enter)
code.bind('<FocusOut>', on_leave)

Frame(frame,width=240,height=2,bg='black').place(x=45,y=185)

#button
Button(frame,width=30,pady=7,text='Sign in',bg='#57a1f8',fg= 'white',border=0,command = signin).place(x=35,y=220 )
label = Label(frame, text = 'Dont have an account?',fg = 'black',bg = 'white', font = ('Microsoft YaHei UI Light',8))
label.place(x= 90,y=270)

sign_up = Button(frame, width = 6, text= 'Sign up', border = 0,bg = 'white', cursor = 'hand2', fg = '#57a1f8')
sign_up.place(x= 225,y= 270)


root.mainloop()