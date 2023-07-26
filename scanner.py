from aiohttp import ClientSession
from llm_client import OpenAIClient, LLMAPIClientConfig
import json
import numpy as np
import cv2
import pytesseract

OPENAI_API_KEY = 'YOUR_OPENAI_APIKEY'

# change depending on OS
# https://ironsoftware.com/csharp/ocr/blog/ocr-tools/tesseract-ocr-windows/#:~:text=To%20test%20that%20Tesseract%20OCR,explanation%20of%20Tesseract's%20usage%20options.&text=Congratulations!,for%20Windows%20on%20your%20machine.
# ^ to find where your tesseract is
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def getText(byte_data, apply_filter=False):  # ocr
    image = cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if apply_filter:
        threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)

        text = pytesseract.image_to_string(threshold)
        filtered_image = threshold
    else:
        text = pytesseract.image_to_string(gray)
        filtered_image = gray
        if 'INGREDIENTS'.lower() not in text.lower():
            return None, filtered_image

    return text, filtered_image


def netify(data, indent=0):  # shows english data
    result = ""
    indent_str = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            result += f"{indent_str}{key}:\n{netify(value, indent + 2)}"
    elif isinstance(data, list):
        for item in data:
            result += netify(item, indent)
    else:
        result += f"{indent_str}{data}\n"
    return result


correctNorgnPrompt = 'When I give you a prompt, you should only the get the ingredients in the text and show each ingredient separated by "+" and nothing else. Your first prompt is: Ingredients: '

IngrListprompt = '''i will give an ingredient list (python) on chat (['ingredientName1', 'ingredientName2']), for each ingredient give the list of pros and cons, at last add a summary of all the ingredients combined and a healthRate on how healthy the combined ingredients are out of 10. USE YOUR KNOWLEDGE AND FILL. Do not add anything else. Example template:
{ [
        "ingredients": [
            "ingredientName1": {
                "pros": [], #list of benifits of the ingredient
                "cons": [] #list of side affects of the ingredient
            },
            "ingredientName2": {
                "pros": [], #list of benifits of the ingredient
                "cons": [] #list of side affects of the ingredient
            }
        ],
        "summary": "", #weather I should use this combination of ingredients
        "healthRate": 0  #out of 10, tell an int on how healthy the combination of ingredients is
    ]
}

and your first list is:
'''


async def main(image_path, apply_filter=False) -> str:
    async with ClientSession() as session:
        llm_client = OpenAIClient(LLMAPIClientConfig(
            OPENAI_API_KEY, session, default_model="text-davinci-003"))

        text = getText(image_path, apply_filter)

        if not text:
            raise ValueError('No ingredients text found, or unclear')

        x = await llm_client.text_completion('what ever is in ``` you must list down the ingredients given in the ingredients section (ingredients:) separated with commas and without a fullstop\n```'+text+'```\n')
        ingredientList = x[0].split(', ')
        y = await llm_client.text_completion(str(ingredientList)+' is a list of ingredients. make a json of pros and cons of each ingredient, and at the end add a summary of the ingredients combined, and a health rate, which out of 10 how healthy the combination of all the products is.'+'\n', max_tokens=1000)
        data = json.loads(y[0])
        return netify(data)
