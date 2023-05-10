import random, json, pickle, string, sys, time, requests, re, os, cv2, random
from bs4 import BeautifulSoup as bs 
from termcolor import colored
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from datetime import date, datetime
import billboard
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()

data = json.loads(open(file='./NLP_Projects/ChatBot/intents.json').read())
file = open("./NLP_Projects/ChatBot/vocab.txt","r")
text = file.read()
voc = text.split(" ")
classes = pickle.load(open('./NLP_Projects/ChatBot/file_classes.pkl','rb'))
words = pickle.load(open('./NLP_Projects/ChatBot/file_words.pkl','rb'))
model = load_model('./NLP_Projects/ChatBot/chatbot.h5')

letters = string.ascii_letters
numbers = string.digits
punctuation = "!#$%+-*<>@&"
list = [letters, numbers, punctuation]

base_url = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = ""

def word_permutation(string):
    letters = [letter for letter in string]
    k = len(string) // 2
    j = -1
    for i in range(k-1) :
        letters[i] , letters[i+1] = letters[i+1], letters[i]
        letters[j], letters[j-1] = letters[j-1], letters[j]
        j = j-1
    new_word = "".join(letters)
    return new_word

def kelvin_to_celsius(kelvin):
    celsius = kelvin - 273.15
    return celsius

def metersec_to_kmh(metersec):
    kmh = metersec * 3.6
    return kmh

def write(text):
    for w in text:
        sys.stdout.write(colored(w,'red'))
        sys.stdout.flush() 
        time.sleep(0.1)

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words =  clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]),verbose = 0)[0]
    threshold = 0.25
    results = [ [i,r] for i,r in enumerate(res) if r > threshold ]
    results.sort(key = lambda x : x[1], reverse = True )
    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]],'probability' : str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
   
    if tag == 'datetime':
        today = date.today()
        return f"Today's date is {today}."

    if tag == 'hourtime':
        time = datetime.now()
        currentTime = time.strftime("%H:%M")
        return f"It's {currentTime}."
    
    if tag == "music":
        chart = billboard.ChartData('hot-100')
        write("Yes, here are the top 10 songs at the moment :\n")
        ls = []
        for i in range(10):
            song = chart[i]
            ls.append(song.title + ' - ' + song.artist)
        return "\n".join(ls)
            
    if tag == "password_help":
        password = ""
        write("What's the desired length ? \n")
        password_length = int(input("You: "))
        for i in range(password_length):
            r = random.choice(list)
            char = random.choice(r)
            password  += char 
        return f"Here's a good password : {password}. \nA hard one to memorise, so try to save it somewhere."
    
    if tag == "math_addition":
        tokens = word_tokenize(message)
        numbers = []
        s = 0
        for x in tokens:
            if x.lstrip('-').isdigit() == True:
                numbers.append(x)
        for nbr in numbers:
            nbr = int(nbr)
            s = s + nbr 
        return f"Its {s}."
    
    if tag == "weather_data":
        sep  = "in "
        sent = message.split(sep, 1)[1]
        city = re.sub(r'[^\w\s]', '', sent)
        city = city.title()
        url  = base_url + "appid=" + API_KEY + "&q=" + city
        response = requests.get(url).json()
        try:
            temp_kelvin = response['main']['temp']
            temp_kelvin_feels_like = response['main']['feels_like']
            humidity = response['main']['humidity']
            wind_speed = response['wind']['speed']
            description = response['weather'][0]['description']
            temp = kelvin_to_celsius(temp_kelvin)
            temp_feels_like = kelvin_to_celsius(temp_kelvin_feels_like)
            wind = metersec_to_kmh(wind_speed)
            return f'''Temperature in {city} is {int(temp)}°C.\nFeels like {int(temp_feels_like)}°C with a {description}.\nHumidity is at {humidity}%.\nWind speed is at {int(wind)}km/h.'''
        except:
            return f"Can't find any weather data about the city {city}."
    
    if tag == "celebrity_data_scrape":
        sep = "called "
        keyword = message.split(sep, 1)[1]
        keyword = re.sub(r'[^\w\s]', '', keyword).title()
        url1 = 'https://en.wikipedia.org/wiki/' + keyword
        page = requests.get(url1)
        soup = bs(page.text, 'html.parser')
        try :
            all_parag = soup.find_all('p')
            for i in range(10):
                parag = all_parag[i]
                text = parag.text
                if text.isspace() or len(text) < 100:
                    pass
                else:
                    text = re.sub("\[.*?\]","",text)
                    text = text.strip()
                    return text
        except :
            return f"Can't find any information about {keyword}."
        
    if tag == "image_data_scrape":
        sep2 = "of "
        name = message.split(sep2, 1)[1]
        url2 = "https://www.google.fr/images?q=" + name 
        page2 = requests.get(url2)
        soup2 = bs(page2.content, "html.parser")
        imgs = soup2.find_all("img")[1:11]
        a = random.randint(1,10)
        image = imgs[a]
        link = image.get("src")
        img_data = requests.get(link).content
        with open(name +'.jpg', 'wb') as handler: 
            handler.write(img_data)
        img = cv2.imread(name+'.jpg')
        img = cv2.resize(img,(250,250))
        write("Loading ...\n")
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", img.shape[1], img.shape[0])
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        if cv2.getWindowProperty('Image',cv2.WND_PROP_VISIBLE) < 1:        
            os.remove(name+'.jpg')
        cv2.destroyAllWindows()
        return f"That was one of the images of {name}."
    
    if tag == "word_game":
        r = random.randint(0, len(voc)-1)
        word = voc[r]
        new_word = word_permutation(word)
        write("Guess the correct spelling of the following word : ")  
        write(new_word)
        answer = input("\nYou: ")
        while(answer != word):
            if (len(answer) > 15) or (" " in answer):
                write("The answer is : ")
                write(word)
                break
            write("Wrong answer, try again.")
            answer = input("\nYou: ")
        return "\nokay"
        
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Type quit to exit the program !")

while True:
    message = input("You: ")
    if message.lower() == "quit":
            break
    ints = predict_class(message)
    res = get_response(ints, data)
    write(res)
    print()