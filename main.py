import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model

model = load_model('chatbot_model.h5')
import random

import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
# to use buttons:
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
# import socket_client
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.storage.jsonstore import JsonStore
import sys

kivy.require("1.10.1")

kivy.require("1.10.1")


# This class is an improved version of Label
# Kivy does not provide scrollable label, so we need to create one
class ScrollableLabel(ScrollView):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ScrollView does not allow us to add more than one widget, so we need to trick it
        # by creating a layout and placing two widgets inside it
        # Layout is going to have one collumn and and size_hint_y set to None,
        # so height wo't default to any size (we are going to set it on our own)
        self.layout = GridLayout(cols=1, size_hint_y=None)
        self.add_widget(self.layout)

        # Now we need two wodgets - Label for chat history and 'artificial' widget below
        # so we can scroll to it every new msg and keep new msgs visible
        # We want to enable markup, so we can set colors for example
        self.chat_history = Label(size_hint_y=None, markup=True)
        self.scroll_to_point = Label()

        # We add them to our layout
        self.layout.add_widget(self.chat_history)
        self.layout.add_widget(self.scroll_to_point)

    # Methos called externally to add new msg to the chat history
    def update_chat_history(self, msg):
        # First add new line and msg itself
        self.chat_history.text += '\n' + msg

        # Set layout height to whatever height of chat history text is + 15 pixels
        # (adds a bit of space at teh bottom)
        # Set chat history label to whatever height of chat history text is
        # Set width of chat history text to 98 of the label width (adds small margins)
        self.layout.height = self.chat_history.texture_size[1] + 15
        self.chat_history.height = self.chat_history.texture_size[1]
        self.chat_history.text_size = (self.chat_history.width * 0.98, None)

        # As we are updating above, text height, so also label and layout height are going to be bigger
        # than the area we have for this widget. ScrollView is going to add a scroll, but won't
        # scroll to the botton, nor there is a method that can do that.
        # That's why we want additional, empty wodget below whole text - just to be able to scroll to it,
        # so scroll to the bottom of the layout
        self.scroll_to(self.scroll_to_point)


class ChatPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We are going to use 1 column and 2 rows
        self.cols = 1
        self.rows = 2

        # First row is going to be occupied by our scrollable label
        # We want it be take 90% of app height
        self.history = ScrollableLabel(height=Window.size[1] * 0.9, size_hint_y=None)
        self.add_widget(self.history)

        # In the second row, we want to have input fields and Send button
        # Input field should take 80% of window width
        # We also want to bind button click to send_msg method
        self.new_msg = TextInput(width=Window.size[0] * 0.8, size_hint_x=None, multiline=False)
        self.send = Button(text="Send")
        self.send.bind(on_press=self.send_msg)

        # To be able to add 2 widgets into a layout with just one collumn, we use additional layout,
        # add widgets there, then add this layout to main layout as second row
        bottom_line = GridLayout(cols=2)
        bottom_line.add_widget(self.new_msg)
        bottom_line.add_widget(self.send)
        self.add_widget(bottom_line)

        # To be able to send msg on Enter key, we want to listen to keypresses
        Window.bind(on_key_down=self.on_key_down)

        # We also want to focus on our text input field
        # Kivy by default takes focus out out of it once we are sending msg
        # The problem here is that 'self.new_msg.focus = True' does not work when called directly,
        # so we have to schedule it to be called in one second
        # The other problem is that schedule_once() have no ability to pass any parameters, so we have
        # to create and call a function that takes no parameters
        Clock.schedule_once(self.focus_text_input, 1)

        # And now, as we have out layout ready and everything set, we can start listening for incimmong msgs
        # Listening method is going to call a callback method to update chat history with new msgs,
        # so we have to start listening for new msgs after we create this layout

    # Gets called on key press
    def on_key_down(self, instance, keyboard, keycode, text, modifiers):

        # But we want to take an action only when Enter key is being pressed, and send a msg
        if keycode == 40:
            self.send_msg(None)

    # Gets called when either Send button or Enter key is being pressed
    # (kivy passes button object here as well, but we don;t care about it)
    def send_msg(self, _):

        # Get msg text and clear msg input field
        msg = self.new_msg.text
        self.new_msg.text = ''

        # If there is any msg - add it to chat history and send to the server
        if msg:
            # Our msgs - use red color for the name
            self.history.update_chat_history(f'[color=dd2020][/color] you> {msg}')



        intents = JsonStore('intents.json')
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))

        def clean_up_sentence(sentence):
            # tokenize the pattern - split words into array
            sentence_words = nltk.word_tokenize(sentence)
            # stem each word - create short form for word
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            return sentence_words

        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

        def bow(sentence, words, show_details=True):
            # tokenize the pattern
            sentence_words = clean_up_sentence(sentence)
            # bag of words - matrix of N words, vocabulary matrix
            bag = [0] * len(words)
            for s in sentence_words:
                for i, w in enumerate(words):
                    if w == s:
                        # assign 1 if current word is in the vocabulary position
                        bag[i] = 1
                        if show_details:
                            print("found in bag: %s" % w)
            return (np.array(bag))

        def predict_class(sentence, model):
            # filter out predictions below a threshold
            p = bow(sentence, words, show_details=False)
            res = model.predict(np.array([p]))[0]
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            # sort by strength of probability
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            return return_list

        def getResponse(ints, intents_json):
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if (i['tag'] == tag):
                    result = random.choice(i['responses'])
                    break
            return result

        def chatbot_response(msg):
            ints = predict_class(msg, model)
            res = getResponse(ints, intents)
            return res

        res = chatbot_response(msg)
        self.history.update_chat_history(f'[color=20dd20][/color] ramu> {res}')

        # As mentioned above, we have to shedule for refocusing to input field
        Clock.schedule_once(self.focus_text_input, 0.1)

    # Sets focus to text input field
    def focus_text_input(self, _):
        self.new_msg.focus = True

    # Passed to sockets client, get's called on new msg
    def incoming_msg(self, res):
        # Update chat history with username and msg, green color for username
        res = chatbot_response(msg)
        self.history.update_chat_history(f'[color=20dd20][/color] ramu> {res}')


class InfoPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Just one column
        self.cols = 1

        # And one label with bigger font and centered text
        self.msg = Label(halign="center", valign="middle", font_size=30)

        # By default every widget returns it's side as [100, 100], it gets finally resized,
        # but we have to listen for size change to get a new one
        # more: https://github.com/kivy/kivy/issues/1044
        self.msg.bind(width=self.update_text_width)

        # Add text widget to the layout
        self.add_widget(self.msg)

    # Called with a msg, to update msg text in widget
    def update_info(self, msg):
        self.msg.text = msg

    # Called on label width update, so we can set text width properly - to 90% of label width
    def update_text_width(self, *_):
        self.msg.text_size = (self.msg.width * 0.9, None)


class EpicApp(App):
    def build(self):
        return ChatPage()

    def create_chat_page(self):
        self.chat_page = ChatPage()
        screen = Screen(name='Chat')
        screen.add_widget(self.chat_page)
        self.screen_manager.add_widget(screen)


# Error callback function, used by sockets client
# Updates info page with an error msg, shows msg and schedules exit in 10 seconds
# time.sleep() won't work here - will block Kivy and page with error msg won't show up
def show_error(msg):
    chat_app.info_page.update_info(msg)
    chat_app.screen_manager.current = 'Info'
    Clock.schedule_once(sys.exit, 10)


if __name__ == "__main__":
    chat_app = EpicApp()
    chat_app.run()



