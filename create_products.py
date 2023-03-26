from typing import Iterator, Literal, Optional
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from ice.cache import diskcache

import re
from pydantic import BaseModel
import requests
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--url", type=str, help="URL of the webpage to parse")
parser.add_argument("--output", type=str, help="Path to output file")

args = parser.parse_args()


TEMPLATE = """I'm working with non-profit. I'm trying to build a merchandise store for them. Below is raw text from their website.

{raw_text}

Write a description of a good merch store to raise funds for this non-profit.

The descriptions should include a general design for the shop. Your description should also include a detailed description of at least 5 different products to be sold. The products that could be created are T-shirt, Hoodie, Backpack, Mug, Beanie, Phone case, Leggings, Jackets, Shoes, Stickers, Tote Bags, Posters, Water Bottles, Greeting Cards and Airpod cases
The products should be in the following format:

Name: [Product name (one of the above)]
Description: [detailed summary of the product, which entices the user to buy it.]
Price: [in cents, e.g. 10000 for $10.00]
Alt-text for design: [description of a pattern for the product design. **Include the style at the end.**, e.g. "a black and white pattern of a cartoon penguin, digital art style". Don't include text that should be written on the product in the alt-text.]

Note: The artist creating the designs is a non-english speaker, so the alt-text should not contain references to quotes or text.

Before giving your description, write your reasoning for what would make the most money for the non-profit. After your description, write the main content section of the website.

The main content section should have the following format:

Header: [Tagline for the website]
Description: [Description of the merchandise store, which entices the user to buy the products.]
Primary color: [CSS color code, e.g. #000000]
Secondary color: [CSS color code, e.g. #000000]
Accent color: [CSS color code, e.g. #000000]

Your answer should be structured as follows:

Reasoning: [Your reasoning for what would the most money for the non-profit, giving a step by step explanation of your thought process and why you think it will maximize revenue]

Description of the website: [Your description of a good merch store to raise funds for this non-profit]

Main content section: [Your description of the main content section of the website in the format above]

Products: [Your description of at least 8 different products to be sold]"""

OPENAI_API_KEY = None

assert OPENAI_API_KEY is not None, "Please set your OPENAI_API_KEY in variable above."

@diskcache()
def run_dalle2(prompt: str) -> str:
    request = requests.post(
        "https://api.openai.com/v1/images/generations",
        json = {
            "prompt": prompt,
            "n": 1,
            "size": "512x512"
        },
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
    )
    return request.json()["data"][0]["url"]


chat = ChatOpenAI(temperature=0.7, model_name="gpt-4", openai_api_key=OPENAI_API_KEY, max_tokens=2000, request_timeout=120)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

#url = "https://support.worldwildlife.org/site/Donation2?df_id=14430&14430.donation=form1&s_src=AWE1800OQ18690A01685RX&gclid=Cj0KCQjwt_qgBhDFARIsABcDjOfelol340VVLXt3XxkkiWLsG2W-OeFJUGc1YdJgTm8aWNqfCXbdBT4aAhU5EALw_wcB"

#https://engage.us.greenpeace.org/onlineactions/_qrOke0FgEmmZdag94x2Jw2?utm_source=gs&utm_medium=ads&utm_content=FD_GS_FR_FY22_Whales_12x_NB&utm_campaign=Inc__220603_FD_GSFROCN_NonBrAJZZZZZZAACZ&sourceid=1013968&ms=FD_GS_FR_FY22_Whales_12x_NB&r=true&am=25&gclid=Cj0KCQjwt_qgBhDFARIsABcDjOegx0mndiAEWF1yRy9zV9eGo4yagXEWt_nYrhkYYkoNelV5ZGGLiVAaAgSSEALw_wcB"

url = args.url

@diskcache()
def get_raw_text(url):
    html = urllib.request.urlopen(url).read()
    return text_from_html(html)

raw_text = get_raw_text(url)

print(raw_text)

print("Parsed text...")

system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant who is very good at interpreting text from raw HTML.")
human_message_prompt = HumanMessagePromptTemplate.from_template(TEMPLATE)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
# get a chat completion from the formatted messages

@diskcache()
def get_raw_response(raw_text, TEMPLATE):
    return chain.run(raw_text=raw_text)

raw_text = get_raw_response(raw_text, TEMPLATE)

PRODUCT_TYPES = ["T-shirt", "Hoodie", "Backpack", "Mug", "Beanie", "Phone case", "Legging", "Jacket", "Shoe", "Sticker", "Tote Bag", "Poster", "Water Bottle", "Greeting Card", "Airpod case"]

class Product(BaseModel):
    name: str
    description: str
    design_alt_text: str
    price: int
    product_type: Literal["T-shirt", "Hoodie", "Backpack", "Mug", "Beanie", "Phone case", "Legging", "Jacket", "Shoe", "Sticker", "Tote Bag", "Poster", "Water Bottle", "Greeting Card", "Airpod case"]
    image: Optional[str] = None

class WebsiteContent(BaseModel):
    header: str
    description: str
    primary_color: str
    secondary_color: str
    accent_color: str

def _parse_product_type(name: str):
    for product_type in PRODUCT_TYPES:
        if product_type.lower() in name.lower():
            return product_type
    return None

def parse_products(product_str: str) -> Iterator[Product]:
    # Use regex to split the file into individual product entries
    pattern = re.compile(r'Name: (.+?)\nDescription: (.+?)\nPrice: (.+?)\nAlt-text for design: (.+?)\n\n', re.DOTALL)
    matches = pattern.findall(product_str)

    # Create a list of Product objects from the matched data
    for match in matches:
        name, description, price, design_alt_text = match
        price = int(price.strip())*10 if price.strip().isdigit() else 10000
        product_type = _parse_product_type(name)
        if product_type is not None:
            yield Product(name=name, description=description, design_alt_text=design_alt_text, price=price, product_type=product_type)

def parse_website_content(main_content: str) -> WebsiteContent:
    pattern = re.compile(r'Header: (.+?)\nDescription: (.+?)\nPrimary color: (.+?)\nSecondary color: (.+?)\nAccent color: (.+?)\n', re.DOTALL)
    match = pattern.findall(main_content)[0]
    header, description, primary_color, secondary_color, accent_color = match
    return WebsiteContent(
        header=header, 
        description=description, 
        primary_color=primary_color, 
        secondary_color=secondary_color, 
        accent_color=accent_color
    )
    


def dalle2_prompt(product: Product):
    return f"A high quality photo of a {product.name.lower()} with a {product.design_alt_text.lower()}"

products_str = raw_text.split("Products:")[1]

products = list(parse_products(products_str))

for product in tqdm(products):
    product.image = run_dalle2(dalle2_prompt(product))

description = raw_text.split("Description of the website:")[1].split("Main content section:")[0].strip()

main_content = raw_text.split("Main content section:")[1].split("Products:")[0]

website_content = parse_website_content(main_content)


with open(args.output, "w") as f:
    json.dump({
        "content":
        {
            "header": website_content.header,
            "description": website_content.description,
            "theme": {
                "primary_color": website_content.primary_color,
                "secondary_color": website_content.secondary_color,
                "accent_color": website_content.accent_color
            }
        },
        "products": [
            {
                "name": product.name,
                "description": product.description,
                "price": product.price,
                "image": product.image,
            }
            for product in products
        ]
    }, f)
    print("Saved..")

