from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from rag.config import EFS_DIR

import re


def extract_text_from_section(section, uri):
    texts = []
    sentence_list = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            if elem.strip():
                texts.append(elem.strip())
        elif re.match("^h", elem.name) != None and len(texts) > 0:
            title = elem.get("id")
            sentence_list.append({"source":f"{uri}#{title}", "text":"\n".join(texts)})
            texts.clear
        else:
            texts.append(elem.get_text().strip())
    return sentence_list


def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]


def extract_sections(record):
    with open(record["path"], "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    sections = soup.find_all(name="div", attrs={"class":"theme-default-content content__default"})
    section_list = []
    uri = path_to_uri(path=record["path"])
    for section in sections:
        section_list += extract_text_from_section(section, uri)
    return section_list


def fetch_text(uri):
    url, anchor = uri.split("#") if "#" in uri else (uri, None)
    file_path = Path(EFS_DIR, url.split("https://")[-1])
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    if anchor:
        target_element = soup.find(id=anchor)
        if target_element:
            text = target_element.get_text()
        else:
            return fetch_text(uri=url)
    else:
        text = soup.get_text()
    return text
