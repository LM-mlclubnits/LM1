import streamlit as st
import torch

from model.tokenizer import Tokenizer
from model.Decoder import Decoder
from model.TrainModel import train_model


st.title("Sentence Completer")

texts = [
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

stories = [
    # A fantasy adventure
    """Once upon a time, in a kingdom surrounded by tall mountains and endless rivers, 
    there lived a young girl named Elara who dreamed of exploring the world beyond her village. 
    Every morning she would stand by the old stone bridge, watching travelers come and go, 
    carrying stories of distant lands filled with mysteries and treasures. 
    One day, Elara discovered a golden feather lying on her windowsill. 
    She picked it up, and suddenly, a voice whispered in her mind: 
    'Follow the feather, and you will find the truth of your destiny.' 
    With courage in her heart, she packed a small bag, said goodbye to her family, 
    and set out on a journey that would take her across forests, deserts, and seas. 
    Along the way, she befriended a talking wolf, saved a village from a terrible flood, 
    and found a map that glowed whenever she was close to her destiny. 
    At last, after many trials, she reached the edge of the world where the sky touched the ocean, 
    and there she discovered that the feather belonged to a phoenix, 
    a creature of fire and rebirth that guarded the secrets of creation itself. 
    With the phoenix as her guide, Elara learned that her true power was not just in seeking adventure, 
    but in giving hope to those she met on her journey.""",

    # A sci-fi exploration
    """In the year 2450, humanity had finally mastered interstellar travel. 
    Captain Arin commanded the starship Horizon, the first vessel to journey beyond the Milky Way. 
    The crew of explorers, scientists, and engineers had trained their whole lives for this mission. 
    As they crossed into the Andromeda Galaxy, they encountered worlds unlike anything they had ever imagined: 
    planets with oceans of glass, forests of metal, and creatures made of living light. 
    On one such planet, they found ruins of an ancient civilization, 
    with towering stone structures that pulsed faintly as if still alive. 
    Arin touched one of the walls, and suddenly a hologram of a being appeared before him, 
    speaking in a language that felt strangely familiar. 
    Over weeks of study, the crew realized that this long-lost species had once traveled the stars, 
    but vanished after unlocking a dangerous technology that tore their world apart. 
    Faced with this discovery, Captain Arin and his crew had to decide whether to continue deeper into the galaxy 
    in search of knowledge or turn back, carrying a warning to all of humanity. 
    In the end, they chose to move forward, not out of recklessness, 
    but out of a belief that wisdom could be gained from the past without repeating its mistakes. 
    Their story became the foundation of a new age of discovery.""",

    # A magical realism story
    """Every summer, in a small coastal town, the sea would bring gifts to the shore. 
    Sometimes it was a piece of driftwood shaped like a crown, 
    sometimes a jar filled with glowing sand, and once, an entire staircase made of seashells. 
    The townspeople whispered that the ocean was alive, and it listened to their dreams. 
    A boy named Theo, curious and restless, decided one day to ask the sea for an adventure. 
    That night, he dreamed of a door made of water, standing tall on the beach. 
    When he woke and ran to the shore, the door was truly there, rippling like liquid glass. 
    Without hesitation, he stepped through it. 
    On the other side, he found himself in a realm where fish flew through the air like birds 
    and clouds floated beneath his feet like stepping stones. 
    He met a girl made of starlight who told him that the ocean connected every world where dreams were real. 
    Together, they sailed on a ship woven from silver threads, 
    visited kingdoms where time moved backward, 
    and battled shadows that tried to swallow entire skies. 
    When Theo finally returned home, he was no longer just a boy from a coastal town— 
    he carried the magic of a thousand worlds in his heart, 
    and every time he closed his eyes, he could still hear the ocean calling his name.""",

    # A modern drama
    """Maya had always lived in the city, surrounded by noise, lights, and people rushing past one another. 
    But after losing her job and feeling her life unravel, 
    she decided to visit her grandmother’s old cottage in the countryside. 
    At first, the silence unsettled her; there were no car horns, no late-night trains, 
    only the soft rustle of leaves and the occasional call of an owl. 
    Yet as days turned into weeks, Maya began to notice things she had long forgotten: 
    the beauty of a sunrise, the way bread smelled as it baked in the oven, 
    the joy of conversations that weren’t hurried. 
    She started tending the small garden, learning to grow her own food, 
    and even mending the old fence that her grandmother had left behind. 
    Slowly, she realized that her worth was not tied to her career or achievements, 
    but to the love she gave and received. 
    One day, as she sat by the fireplace reading a book, 
    she understood that she wasn’t lost at all—she had simply been searching in the wrong places. 
    The city would always be there, but here, in this quiet corner of the world, 
    Maya found herself again."""
]


tokenizer = Tokenizer(stories)
vocab_size = len(tokenizer.stoi)

model = Decoder(vocab_size)

if st.button("Train"):
    train_model(model, texts, tokenizer, epochs=200)

input_text = st.text_input("Enter some words: ")
if st.button("Generate"):
    if input_text.strip():
        idx = torch.tensor([tokenizer.encode(input_text)], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=40)
        st.write("Generated: ", tokenizer.decode(out[0].tolist()))
