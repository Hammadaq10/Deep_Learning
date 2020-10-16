import chat_functions as func
import tkinter
from tkinter import *

# Create a send button function
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        result = func.respose_chatbot(msg)
        ChatLog.insert(END,"Bot: " + result + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("500x600")
base.resizable(width=FALSE,height=TRUE)

# Create ChatBot Window
ChatLog = Text(base,bd=0,bg="white",height="8",width="60",font="Helvetica")
ChatLog.config(state=DISABLED)

# Made Scrollbar into the chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor= "heart")
ChatLog['yscrollcommand'] = scrollbar.set

#  Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send)

# Create Entry Box
EntryBox= Text(base, bd=0, bg="white",width="29", height="5", font="Helvetica")

# Place components on screen
scrollbar.place(x=400,y=6,height=450)
ChatLog.place(x=6,y=6,height=450,width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()

