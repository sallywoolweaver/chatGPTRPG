import requests
import pyttsx3
from gtts import gTTS
from dalleimages import Dalle
import threading
import pygame

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("API_KEY")

def generate_story():
    response = requests.post(
        "https://api.openai.com/v1/engines/text-davinci-002/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": PROMPT,
            "max_tokens": 1024,
            "n": 1,
            "stop": None,
            "temperature": 0.5,
        },
    )
    response_json = response.json()
    return response_json["choices"][0]["text"]

def summarize_story(story):
    response = requests.post(
        "https://api.openai.com/v1/engines/text-davinci-002/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
             "prompt": "Summerize in 15 words this story: "+story,
            "max_tokens": 1024,
            "n": 1,
            "stop": None,
            "temperature": 0.5,
        }
    )
    response_json = response.json()
    return response_json["choices"][0]["text"]

def save_story(story):
    with open("story.txt", "w") as file:
        file.write(story)
    
def dis_story(summarized_story):
    Dalle(summarized_story)   
    
def speak_textPYTT(story):
    engine = pyttsx3.init()
    engine.say(story)
    engine.runAndWait()
    #print('finished talking')

def speak_text(story):
    tts = gTTS(text=story, lang='en', slow=False)
    tts.save("story.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("story.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)

    
if __name__ == "__main__":
    PROMPT = "Generate a long bedtime story. Make it long. Give the main character a name. Make it educational. It should be PG-13. It should be set in:"
    print("Please choose a story prompt:")
    print("1. Fantasy")
    print("2. Post-apocalyptic")
    print("3. Science fiction")
    print("4. Historical")
    print("5. Dystopian")
    print("6. Mythical")
    print("7. Politics")
    print("8. Superhero")
    print("9. Cyberpunk")
    print("10. Mystery")
    prompt_choice = (int)(input("Enter the number of your choice:\n"))
    if(prompt_choice==1):
        PROMPT += 'A story set in a world of magic, mythical creatures, and ancient civilizations, with a chosen hero on a quest to save the world from an evil force.'
    elif(prompt_choice==2):
        PROMPT += 'A story set in a world that has been destroyed by a catastrophic event, with the player character navigating a harsh, new world filled with danger and uncertainty.'   
    elif(prompt_choice==3):
        PROMPT +='A story set in a futuristic world filled with advanced technology, interstellar travel, and otherworldly beings.'
    elif(prompt_choice==4):
        PROMPT += ' story set in a specific time and place in history, allowing the player to explore a different culture and way of life while also engaging in battles and completing quests. It should be set in a factual setting on Earth.'
    elif(prompt_choice==5):
        PROMPT += 'A story set in a dystopian society, where the player character must navigate through a totalitarian government and rebel against it to bring freedom to the people.'
    elif(prompt_choice==6):
        PROMPT += 'A story set in a world of mythical creatures and legends, where the player character must collect powerful artifacts and complete challenges to become a legend themselves.'
    elif(prompt_choice==7):
        PROMPT += 'A story set in a world of high stakes political intrigue, where the player character must gather information, make alliances, and engage in battles of wit to achieve their ultimate goal.'
    elif(prompt_choice==8):
        PROMPT += 'A story set in a world of superheroes and villains, where the player character must use their powers and abilities to stop the world\'s greatest threats and become the greatest hero of all time.'
    elif(prompt_choice==9):
        PROMPT += 'A story set in a post-modern world of technology, where the player character must navigate the dangers of cyberspace and stop a cyber-terrorist from destroying the world\'s infrastructure.'
    elif(prompt_choice==10):
        PROMPT += 'A mystery set in a small, isolated town where strange events have been happening and the player character is tasked with solving the puzzles and uncovering the truth behind the mysterious happenings before it\'s too late.'
    else:
        print("Error\n")
    story = generate_story()
    print(story)
    summarized_story = summarize_story(story)
    text_thread = threading.Thread(target=speak_text, args=(story,))
    text_thread.start()
    save_story(story)
    dis_story(summarized_story)
    text_thread.join()
