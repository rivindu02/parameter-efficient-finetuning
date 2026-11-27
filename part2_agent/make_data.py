"""Template-based dataset generator for the tool-calling agent.

Train and test use *disjoint* phrasing templates (and a different date format
for the date task), so the test set measures generalisation to new wording.

Each record:
  id, split, category, query,
  gold_calls: list of {"name", "arguments"} in order,
  checks: list of {"type": "num"|"str", "value"} the final answer must satisfy,
  messages: full gold conversation (user / assistant tool_calls / tool / assistant)
"""
import argparse
import json
import os
import random

from tools import STOCKS, WEATHER, execute_tool, tool_result_str

SEED = 42
CITIES = [c.title() for c in WEATHER]
TICKERS = list(STOCKS)

# ----------------------------------------------------------------------------
# Phrasing templates: TRAIN and TEST pools never share a template.
# ----------------------------------------------------------------------------
T = {
    "weather": {
        "train": ["What's the weather like in {city} right now?",
                  "How is the weather in {city}?",
                  "Tell me the current weather in {city}.",
                  "Is it hot or cold in {city} today?",
                  "weather {city}",
                  "Can you check the weather for {city}?"],
        "test": ["I'm flying to {city} tonight, what conditions should I expect there?",
                 "Give me a quick weather report for {city}, please.",
                 "Any idea what it's like outside in {city} at the moment?",
                 "Look up the current temperature and sky conditions in {city}."],
    },
    "stock": {
        "train": ["What is the current price of {ticker}?",
                  "How much is {ticker} stock trading at?",
                  "Get me the share price for {ticker}.",
                  "{ticker} price?",
                  "Check the stock price of {ticker} for me."],
        "test": ["What's one share of {ticker} going for these days?",
                 "I want to know the latest quote on the {ticker} ticker.",
                 "Could you pull up where {ticker} is trading?",
                 "Before I invest, tell me what {ticker} costs per share."],
    },
    "calc": {
        "train": [("What is {a} times {b}?", "{a} * {b}"),
                  ("Calculate {a} plus {b}.", "{a} + {b}"),
                  ("What's {a} minus {b}?", "{a} - {b}"),
                  ("Divide {a} by {b}.", "{a} / {b}"),
                  ("Compute ({a} + {b}) * {c}.", "({a} + {b}) * {c}"),
                  ("What is {a} squared?", "{a} ** 2")],
        "test": [("Multiply {a} by {b} for me.", "{a} * {b}"),
                 ("If I add {a} and {b} and then multiply the total by {c}, what do I get?", "({a} + {b}) * {c}"),
                 ("Take {b} away from {a}. What's left?", "{a} - {b}"),
                 ("Split {a} into {b} equal parts; how big is each part?", "{a} / {b}"),
                 ("What do you get when you raise {a} to the power of 3?", "{a} ** 3")],
    },
    "convert": {
        "train": ["Convert {v} {fu} to {tu}.",
                  "How many {tu} is {v} {fu}?",
                  "What is {v} {fu} in {tu}?",
                  "{v} {fu} to {tu}"],
        "test": ["My recipe says {v} {fu}; how much is that in {tu}?",
                 "I ran {v} {fu} this morning, roughly how many {tu} is that?",
                 "Express {v} {fu} using {tu} instead.",
                 "Change {v} {fu} over to {tu}, please."],
    },
    "days": {
        "train": ["How many days are there between {d1} and {d2}?",
                  "Count the days from {d1} to {d2}.",
                  "Number of days from {d1} until {d2}?",
                  "How long is it in days from {d1} to {d2}?"],
        "test": ["My trip starts on {d1} and ends on {d2}. How many days is that?",
                 "If today is {d1}, how many days until {d2}?",
                 "Work out the gap in days between {d1} and {d2}."],
    },
    "weather_f": {
        "train": ["What's the temperature in {city} in Fahrenheit?",
                  "How warm is it in {city} in degrees Fahrenheit?",
                  "Tell me the temperature in {city} converted to Fahrenheit."],
        "test": ["My American friend wants to know how hot it is in {city}, in Fahrenheit.",
                 "Give me {city}'s current temperature but in °F."],
    },
    "stock_shares": {
        "train": ["How much would {n} shares of {ticker} cost?",
                  "What's the total cost of buying {n} shares of {ticker}?",
                  "I want {n} shares of {ticker}. How much will that be?"],
        "test": ["If I bought {n} {ticker} shares at today's price, what would I pay in total?",
                 "Price out a purchase of {n} shares of {ticker} for me."],
    },
}

# Unit pairs the user may ask about: (spoken from-unit, spoken to-unit, canonical from, canonical to)
UNIT_PAIRS = [
    ("kilometers", "miles", "km", "mi"), ("miles", "kilometers", "mi", "km"),
    ("kg", "pounds", "kg", "lb"), ("pounds", "kilograms", "lb", "kg"),
    ("meters", "feet", "m", "ft"), ("feet", "meters", "ft", "m"),
    ("liters", "gallons", "L", "gal"), ("gallons", "liters", "gal", "L"),
    ("cm", "inches", "cm", "in"), ("inches", "centimeters", "in", "cm"),
    ("degrees Celsius", "Fahrenheit", "C", "F"), ("degrees Fahrenheit", "Celsius", "F", "C"),
]

# No-tool requests: (query, answer used as training target, keyword the answer must contain or None)
NO_TOOL = {
    "train": [
        ("What is the capital of France?", "The capital of France is Paris.", "paris"),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.", "shakespeare"),
        ("How many continents are there?", "There are 7 continents.", "7"),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.", "au"),
        ("What planet is known as the Red Planet?", "Mars is known as the Red Planet.", "mars"),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth.", "pacific"),
        ("How many legs does a spider have?", "A spider has 8 legs.", "8"),
        ("What language is spoken in Brazil?", "The main language spoken in Brazil is Portuguese.", "portuguese"),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci.", "leonardo"),
        ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide (CO2) from the air.", "carbon dioxide"),
        ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level.", "100"),
        ("How many days are in a week?", "There are 7 days in a week.", "7"),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world.", "everest"),
        ("Translate 'thank you' into Spanish.", "'Thank you' in Spanish is 'gracias'.", "gracias"),
        ("What does CPU stand for?", "CPU stands for Central Processing Unit.", "central processing unit"),
        ("What is a stock market?", "A stock market is a place where shares of public companies are bought and sold.", "shares"),
        ("Why is the sky blue?", "The sky looks blue because air molecules scatter short blue wavelengths of sunlight more than other colours (Rayleigh scattering).", "scatter"),
        ("What causes rain?", "Rain forms when water vapour in clouds condenses into droplets that grow heavy enough to fall.", "water"),
        ("What is a kilometer?", "A kilometer is a unit of length equal to 1000 meters.", "1000"),
        ("Hi there!", "Hello! How can I help you today?", None),
        ("Thanks for your help.", "You're welcome! Let me know if you need anything else.", None),
        ("Write a two-line poem about the sea.", "The sea rolls in with silver light,\nand hums the shore to sleep at night.", None),
        ("Give me a tip for staying focused while studying.", "Try working in 25-minute blocks with short breaks in between, and keep your phone out of reach.", None),
        ("Tell me a fun fact about octopuses.", "Octopuses have three hearts and blue blood.", None),
        ("What's your name?", "I'm a helpful assistant. How can I help you?", None),
        ("Explain what an API is in one sentence.", "An API is a set of rules that lets one piece of software request data or actions from another.", None),
        ("Suggest a name for a pet cat.", "How about 'Mochi'?", None),
        ("What is the opposite of 'ancient'?", "The opposite of 'ancient' is 'modern'.", "modern"),
    ],
    "test": [
        ("Which country is home to the city of Kyoto?", "Kyoto is in Japan.", "japan"),
        ("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity.", "einstein"),
        ("What's the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius.", "0"),
        ("Name the longest river in Africa.", "The Nile is the longest river in Africa.", "nile"),
        ("How many sides does a hexagon have?", "A hexagon has 6 sides.", "6"),
        ("What does 'HTML' stand for?", "HTML stands for HyperText Markup Language.", "markup"),
        ("Which organ pumps blood around the human body?", "The heart pumps blood around the body.", "heart"),
        ("In general, why do stock prices go up and down?", "Stock prices move with supply and demand, driven by company results, news and investor expectations.", None),
        ("Explain in simple words what a weather forecast is.", "A weather forecast is a prediction of future weather made from current observations and models.", None),
        ("Good morning! How are you doing?", "Good morning! I'm doing well, thanks. How can I help?", None),
        ("Could you write me a short motivational sentence?", "Small steps every day add up to big results.", None),
        ("What is the plural of 'mouse'?", "The plural of 'mouse' is 'mice'.", "mice"),
        ("Recommend a good habit for better sleep.", "Keep a consistent bedtime and avoid screens for an hour before sleep.", None),
        ("What is the main ingredient in guacamole?", "The main ingredient in guacamole is avocado.", "avocado"),
        ("Tell me a short joke.", "Why did the scarecrow win an award? Because he was outstanding in his field.", None),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon.", "armstrong"),
        ("Describe the colour of a ripe banana in one word.", "Yellow.", "yellow"),
        ("What do bees make?", "Bees make honey.", "honey"),
        ("Is a tomato a fruit or a vegetable, botanically?", "Botanically, a tomato is a fruit.", "fruit"),
        ("Say hello in French.", "Hello in French is 'bonjour'.", "bonjour"),
    ],
}

# ----------------------------------------------------------------------------
# Answer formatting (training targets)
# ----------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def fmt_num(x):
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


