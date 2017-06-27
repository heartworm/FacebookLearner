import glob
import json
import re
import os

search_path = "~/messages/" #path with json files output by facebook_dumper.py with trailing '/'

file_list = glob.glob(search_path + "*.json")
file_list.sort(reverse=True, key=lambda f: int(f[:f.find("-")])) # Expects file names of the form <int>-<int>.json

all_letters = "abcdefghijklmnopqrstuvwxyz0123456789 "
symbols_pattern = re.compile(r"[^{}]".format(all_letters))
whitespace_pattern = re.compile(r"\s+")

messages = []
current_message_id = ""
emails = []

def sanitize_string(str_in):
    str_in = str_in.lower() # Change all characters to lowercase
    str_in = symbols_pattern.sub(" ", str_in) # Replace all chars not in all_letters with spaces
    str_in = whitespace_pattern.sub(" ", str_in) # Replace all back-to-back whitespace with single spaces
    str_in = str_in.strip() # Remove whitespace on either side of the string
    return str_in

if not os.path.exists("output"):
    os.mkdir("output")

if __name__ == "__main__":
    with open("messages.json", "w") as out_file:
        for file_name in file_list:
            with open(file_name, "r") as json_file:
                json_obj = json.load(json_file)
                actions = json_obj["payload"]["actions"] # Actions are events that occur (like messages) in the chat.
                for action in actions:
                    message_id = action["message_id"]
                    if message_id == current_message_id:
                        # To serve as a reference to the next message, facebook duplicates messages across XHR requests.
                        continue
                    current_message_id = message_id
                    email = action["author_email"]
                    if action["action_type"] != "ma-type:user-generated-message":
                        # This could be expanded to support stickers, likes, group name changes etc.
                        continue
                    if email not in emails:
                        # Give authors fake names, to be corrected in the output file.
                        emails.append(email)
                    if "body" in action:
                        out_message = sanitize_string(action["body"])
                        # If the string was in chinese for example, out_message would be blank.
                        if out_message is not None and out_message != "":
                            messages.append({
                                "email": email,
                                "message": out_message
                            })
        json.dump({
            "authors": [{"email": email, "name": "human"} for email in emails],
            "letters": all_letters,
            "messages": messages
        }, out_file, indent='  ')