import streamlit as st
import torch

from model.tokenizer import Tokenizer
from model.Decoder import Decoder
from model.TrainModel import train_model
# from data.stories_list import stories  # optionally use external dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Sentence Completer (BPE Tokenizer)")

# small default dataset (you can replace with stories or a larger dataset)
data_texts = [
    "Hello, how are you?",
    "I am fine, thank you.",
    "What are you doing?",
    "I am reading a book.",
    "Do you like coffee?",
    "Yes, I love coffee.",
    "Where are you going?",
    "I am going to the market.",
    "What is your name?",
    "My name is John.",
    "Nice to meet you.",
    "Nice to meet you too.",
    "How was your day?",
    "It was good, thank you.",
    "Did you eat lunch?",
    "Yes, I had rice and vegetables.",
    "What time is it?",
    "It is three o’clock.",
    "Can you help me?",
    "Yes, of course.",
    "I am feeling tired today.",
    "You should take some rest.",
    "Where do you live?",
    "I live near the park.",
    "Do you like music?",
    "Yes, I listen to music every day.",
    "Which is your favorite color?",
    "My favorite color is blue.",
    "What is your favorite food?",
    "I like pizza the most.",
    "Are you busy right now?",
    "Yes, I am working on my computer.",
    "No, I am free. Let’s talk.",
    "Do you have a pet?",
    "Yes, I have a dog.",
    "No, I don’t have any pets.",
    "What are you watching?",
    "I am watching a movie.",
    "Which movie are you watching?",
    "I am watching a comedy film.",
    "Do you like sports?",
    "Yes, I play football with my friends.",
    "No, I don’t like sports much.",
    "What is the weather like today?",
    "It is sunny and warm.",
    "It is raining outside.",
    "Are you coming to the party?",
    "Yes, I will be there.",
    "No, I cannot come today.",
    "Can you tell me the way?",
    "Go straight and turn left.",
    "What did you do yesterday?",
    "I went shopping with my family.",
    "I stayed home and rested.",
    "Are you hungry?",
    "Yes, let’s eat something.",
    "No, I just had lunch.",
    "Do you want some tea?",
    "Yes, please. Thank you.",
    "No, I prefer coffee.",
    "Can I borrow your pen?",
    "Sure, here it is.",
    "Sorry, I don’t have one.",
    "When is your birthday?",
    "My birthday is in July.",
    "Do you like reading books?",
    "Yes, I read every night before bed.",
    "No, I like watching TV instead.",
    "What are your hobbies?",
    "I like drawing and painting.",
    "I like playing video games.",
    "Are you free tomorrow?",
    "Yes, let’s meet at the park.",
    "No, I have some work to do.",
    "Hi, how are you? I’m good, thanks. What about you? I’m fine too.",
    "Hello, are you busy? Not really. Do you need help? Yes, with my homework.",
    "Good morning! Did you sleep well? Yes, I slept fine. What about you? Same here.",
    "Hey, did you eat lunch? Not yet. Want to eat together? Sure, let’s go.",
    "Hi, what are you doing? Just reading. Do you want to join me? Okay.",
    "Hello, are you coming to class today? Yes, I’ll be there. Don’t be late. I won’t.",
    "Good evening! Did you watch TV? Yes, a movie. Was it good? It was okay.",
    "Hey, can you help me? Of course. What do you need? Just some advice.",
    "Hi, do you want to play a game? Sure, which one? Let’s play chess. Okay.",
    "Hello, is it raining outside? Yes, take an umbrella. Thanks for reminding me.",
    "Hi, where are you going? I’m going to the market. Do you need anything? Maybe some apples.",
    "Hey, did you finish your homework? Not yet. I’ll do it tonight. Don’t forget!",
    "Good morning! What’s for breakfast? Just bread and eggs. Sounds good.",
    "Hello, can I borrow your pen? Sure, here you go. Thanks a lot.",
    "Hi, do you like coffee? Yes, I love it. Do you? Not really, I prefer tea.",
    "Hey, is the bus late again? Yeah, it’s always late. That’s annoying.",
    "Hi, do you want to go for a walk? Sure, let’s go after dinner. Okay.",
    "Hello, what are you reading? Just a novel. Is it interesting? Very much.",
    "Good evening! How was your day? It was fine, a little busy. Same here.",
    "Hi, do you play football? Yes, sometimes. Do you? No, I play cricket.",
    "Hey, what’s your favorite food? I like pizza. How about you? I love pasta.",
    "Hi, can you help me with this bag? Of course, let me carry it. Thanks!",
    "Hello, do you like music? Yes, I listen every day. Me too.",
    "Hi, do you know the time? Yes, it’s 3 PM. Thank you.",
    "Hey, are you free tomorrow? Yes, why? Let’s meet at the park. Sounds good.",
    "Good morning! Do you want some tea? Yes, please. With sugar? Just a little.",
    "Hi, where do you live? I live near the school. Oh, that’s close to me too.",
    "Hello, did you call me? Yes, I wanted to ask something. What is it?",
    "Hey, do you like movies? Yes, I watch on weekends. Same here.",
    "Hi, are you hungry? A little. Want to eat together? Sure."
]

# Initialize or train BPE tokenizer (if tokenizer files don't exist it will train & save)
tokenizer = Tokenizer(texts=data_texts, vocab_size=5000, save_dir="./tokenizer")
vocab_size = tokenizer.get_vocab_size()

# Initialize model and move to device
model = Decoder(vocab_size).to(device)

# Train button
if st.button("Train"):
    train_model(model, data_texts, tokenizer, epochs=200)

# Generation controls
input_text = st.text_input("Enter some words: ")
max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=512, value=100, step=10)
if st.button("Generate"):
    if input_text.strip():
        idx = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        generated_text = tokenizer.decode(out[0].tolist())
        st.write("Generated: ", generated_text)
