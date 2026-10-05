import requests
from bs4 import BeautifulSoup
import csv

url = "https://books.toscrape.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find("div", class_="side_categories").find_all("a")

all_books = []
rating_map = {"One": 1.0, "Two": 2.0, "Three": 3.0, "Four": 4.0, "Five": 5.0}

#loop on categories
for link in links[1:]:
    folder = url + link["href"].replace("index.html", "")
    page_url = folder + "index.html"

    #loop in pages
    while page_url:
        response = requests.get(page_url)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        category = soup.find("h1").text
        books = soup.find_all("article", class_="product_pod")

        #loop in books
        for book in books:
            title = book.find("h3").find("a")["title"]

            price_text = book.find("p", class_="price_color").text
            price = float(price_text.replace("£", ""))

            rating_word = book.find("p", class_="star-rating")["class"][1]
            rating = rating_map[rating_word]

            availability_text = book.find("p", class_="instock availability").text
            available = "In stock" in availability_text

            image = book.find("img")["src"]
            image = url + image.replace("../", "")

            all_books.append({
                "title": title,
                "price": price,
                "rating": rating,
                "available": available,
                "category": category,
                "image": image
            })

        next_button = soup.find("li", class_="next")
        if next_button:
            page_url = folder + next_button.find("a")["href"]
        else:
            page_url = None

with open("books.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["title", "price", "rating", "available", "category", "image"])
    writer.writeheader()
    writer.writerows(all_books)

print(len(all_books), "books saved")
