import sys
from PyQt5 import QtWidgets, QtGui, QtCore

from huggingface_hub import login

# Replace with your actual Hugging Face API token
HUGGINGFACE_API_KEY = "YOUR HUGGING FACE API KEY HERE"

# Login to Hugging Face
login(HUGGINGFACE_API_KEY)
print("Logged in to Hugging Face!")

# Loading LLaMA Model 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose device: "cuda" if available, otherwise "cpu"
USE_CUDA = torch.cuda.is_available()  # Set to False to force CPU
device = "cuda" if USE_CUDA else "cpu"

# Load model & tokenizer
model_name = "Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if USE_CUDA else torch.float32,
    device_map="auto" if USE_CUDA else None
)

print(f"Model loaded on: {device}")

# Chat Function
def chat(prompt, max_length=100, temperature=0.9, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,  # Limit response length
            temperature=temperature,  # Controls randomness (lower = more deterministic)
            top_p=top_p,  # Nucleus sampling (lower = more focused)
            do_sample=True,  # Enables randomness in generation
            repetition_penalty=1.2  # Reduces repetitive output
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):] #Remove prompt from response

# Example usage
#print(chat("what is the capital of Nigeria?", max_length=100))

# GUI Interface
class ChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat with LLaMA")
        self.resize(600, 400)

        # Main layout
        self.layout = QtWidgets.QVBoxLayout(self)

        # Chat history area (QTextEdit - Read Only)
        self.chat_history = QtWidgets.QTextEdit()
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)

        # Horizontal layout for user input and send button
        self.input_layout = QtWidgets.QHBoxLayout()

        # Text input (QLineEdit)
        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_layout.addWidget(self.input_field)

        # Send button
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.send_button)

        self.layout.addLayout(self.input_layout)

        # Connect Enter key to send message
        self.input_field.returnPressed.connect(self.send_message)

    def send_message(self):
        user_text = self.input_field.text().strip()
        if user_text:
            # Add user message to chat history
            self.chat_history.append(f"<b>Me:</b> {user_text}")
            
            # Clear input
            self.input_field.clear()
            
            # Disable input during processing
            self.input_field.setDisabled(True)
            self.send_button.setDisabled(True)
            
            
            self.chat_history.append(f"<i>Your bot is typing...</i>")

            # Clear input
            self.input_field.clear()

            # # Process bot response after a slight delay 
            # QtCore.QTimer.singleShot(100, lambda: chat(user_text))

            # Bot reply
            bot_reply = chat(user_text)

            # Remove "Bot is typing..." placeholder
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            self.chat_history.append(f"<b>Your bot:</b> {bot_reply}\n")

            # Re-enable input
            self.input_field.setDisabled(False)
            self.send_button.setDisabled(False)
            self.input_field.setFocus()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
