import tkinter as tk

from utils.MessageNet import MessageNet
from utils.MessageReader import MessageReader


class ChatGUI:
    def __init__(self, master, mn):
        self.max_messages = 10 #how many messages should the network's reply consist of at most
        self.max_length = 500 #how many characters should the network generate waiting for a message <END> token

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
        self.chat_text.tag_config("user_author", background="black", foreground="white")
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
        tk.Grid.columnconfigure(master, 1, minsize=100)
        tk.Grid.rowconfigure(master, 1, weight=1)

    def send_button_pressed(self, *args):
        input_name = self.author_option_var.get()
        input_message_text = self.message_box_var.get()
        self.message_box_var.set("")
        self.disp_message(input_name, input_message_text, True)

        input_author_ind = self.all_names.index(input_name)
        input_msg = {
            "email": self.all_emails[input_author_ind],
            "message": input_message_text
        }

        is_different_author = False
        is_returned_author = False

        index_sequence, hidden = self.mn.sample_message(input_msg, self.max_length, hidden=self.hidden)

        for n in range(self.max_messages):
            for msg in self.mn.mr.index_sequence_to_messages(index_sequence):
                author_ind = self.all_emails.index(msg["email"])

                if author_ind == input_author_ind:
                    if is_different_author:
                        is_returned_author = True
                        break
                else:
                    is_different_author = True

                self.disp_message(self.all_names[author_ind], msg["message"], False)
            if not is_returned_author:
                self.hidden = hidden
                index_sequence, hidden = self.mn.sample(index_sequence[-1:], self.max_length, hidden=self.hidden)
            else:
                break

    def disp_message(self, name, message, is_user_message):
        tags = ("user_author") if is_user_message else tuple()
        self.chat_text.insert(tk.END, "{}: ".format(name), tags)
        self.chat_text.insert(tk.END, "{}\n".format(message))

if __name__ == "__main__":
    mr = MessageReader("messages.json")
    mn = MessageNet(mr)
    if not mn.load_state():
        print("Failed to load Net state. Exiting.")
        exit(1)

    root = tk.Tk()
    chat_gui = ChatGUI(root, mn)
    root.mainloop()
