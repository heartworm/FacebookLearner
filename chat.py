from MessageReader import MessageReader
from MessageNet import MessageNet
import tkinter as tk

class ChatGUI:
    def __init__(self, master, mn):
        self.mn = mn
        self.all_names = mn.mr.all_names
        self.all_emails = mn.mr.all_emails
        self.hidden = None
        self.master = master
        master.title("Chat")

        self.chat_frame = tk.Frame(master, borderwidth=1, relief=tk.SUNKEN)
        self.chat_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S)

        self.chat_scrollbar = tk.Scrollbar(self.chat_frame)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text = tk.Text(self.chat_frame, yscrollcommand=self.chat_scrollbar.set)
        self.chat_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.author_option_var = tk.StringVar()
        self.author_option_var.set(self.all_names[0])
        self.author_option = tk.OptionMenu(master, self.author_option_var, *tuple(self.all_names))
        self.author_option.grid(row=2, column=1, sticky=tk.W+tk.E+tk.N+tk.S)

        self.message_box_var = tk.StringVar()
        self.message_box = tk.Entry(master, textvariable=self.message_box_var)
        self.message_box.bind("<Return>", self.send_button_pressed)
        self.message_box.grid(row=2, column=2, sticky=tk.W + tk.E + tk.N + tk.S)

        self.send_button = tk.Button(master, text="Send", command=self.send_button_pressed)
        self.send_button.grid(row=2, column=3)

        tk.Grid.columnconfigure(master, 2, weight=1)
        tk.Grid.rowconfigure(master, 1, weight=1)

    def send_button_pressed(self, *args):
        input_name = self.author_option_var.get()
        input_message_text = self.message_box_var.get()
        self.disp_message(input_name, input_message_text)

        author_ind = self.all_names.index(input_name)
        input_msg = {
            "email": self.all_emails[author_ind],
            "message": input_message_text
        }
        index_sequence, self.hidden = self.mn.sample(input_msg, 500, hidden=self.hidden)
        for msg in self.mn.mr.index_sequence_to_messages(index_sequence):
            author_ind = self.all_emails.index(msg["email"])
            self.disp_message(self.all_names[author_ind], msg["message"])

    def disp_message(self, name, message):
        self.chat_text.insert(tk.END, "{}: {}\n".format(name, message))

if __name__ == "__main__":
    mr = MessageReader("new_messages.json")
    mn = MessageNet(mr)
    if not mn.load_state():
        print("Failed to load Net state. Exiting.")
        exit(1)

    root = tk.Tk()
    chat_gui = ChatGUI(root, mn)
    root.mainloop()