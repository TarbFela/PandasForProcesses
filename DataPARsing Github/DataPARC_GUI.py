from DataPARsing import *

__dir__ = __file__.split('\\')[:-1]
for i in __dir__[1:]: __dir__[0] += '\\' + str(i)
__dir__ = __dir__[0]

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo, askyesno

# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('300x150')

data = None


    
class action_button:
    def __init__(self, root = root, text='Give me text'):
        self.button = ttk.Button( root, text=text, command=self.command )
        self.button.pack(expand=True)
    def command(self, message = "oh okay that works too"):
        showinfo( title = 'plt_button window', message = message)
        
class xlsx_file_select_button(action_button):
    def __init__(self, root = root):
        super().__init__(root, text = "Select Excel File")
    def command(self):
        fp = select_file("xlsx")
        print(fp)
        if get_confirm("Do you want to import this file from scratch?\nThis may take a while."):
            global data
            data = getData_v2(fp)
        if get_confirm("Would you like to cache this data?\nThis will make it easier to re-load in the future"):
            data_file = open(filename[:-5]+"_"+picklename,'ab')
            pickle.dump( data, data_file)
            data_file.close()
        
        
            
            
def get_confirm(t="Confirm"):
    return askyesno(message=t)


def select_file(type_str = "xlsx"):
    type_str = "*." + type_str
    filetypes = (
        ('text files', type_str),
        ('All files', '*.*')
    )
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=__dir__,
        filetypes=filetypes)

    if len(filename) == 0: filename = "No File Selected"
    showinfo(
        title='Selected File',
        message=filename
    )
    return filename


# open button
open_button = xlsx_file_select_button()

poop = action_button()

open_button.pack(expand=True)


# run the application
root.mainloop()
