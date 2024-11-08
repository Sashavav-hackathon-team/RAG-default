import os
import fitz
from tqdm.auto import tqdm

def make_it_readable(path : str):
    if not os.path.exists(path):
        print("[INFO] File doesn't exists")
        return
    document = open_and_read(path=path)
    return document
    

def preprocessing_text(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read(path: str) -> list[dict]:
    doc = fitz.open(path)
    text_info = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = preprocessing_text(text=text)
        text_info.append({"page_number": page_number,
                         "page_char_count": len(text),
                         "page_word_count": len(text.split(" ")),
                         "page_setence_count_raw": len(text.split(". ")),
                         "page_token_count": len(text) / 4,
                          "text": text })
    return text_info