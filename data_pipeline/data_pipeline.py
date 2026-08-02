import requests
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup as soup

# Website's url Address
base_url = "http://books.toscrape.com/"

# Create a response object to get the web page's HTML content
def get_text(url):
    try:
        response = requests.get(url)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        # create beautiful soup object to parse HTML text with the help of html.parser
        return soup(response.text, "html.parser")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except requests.exceptions.Timeout as e:
        print(f"Request Timed out: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    return None

# Finding category names and its url's
def get_category_links(n_categories = 5):
    """Get first n_categories category names + url's from homepage sidebar."""
    try:
        soup = get_text(base_url)
        if soup is None:
            return []
        # search for every category link inside the sidebar
        sidebar = soup.select("div.side_categories ul li ul li a")
        categories = []
        for category in sidebar[:n_categories]:
            # Get category name by extracting the text part of <a> element
            # Strip the spaces before and after the name
            name = category.text.strip()
            # get the url, which results in product list page under category
            href = category["href"]
            # Complete category url by adding the base url
            full_url = base_url + href
            # Save the information in tuple to the list
            categories.append((name, full_url))
        return categories

    except AttributeError as e:
        print(f"HTML Structure has changed: {e}")
        return []

    except KeyError as e:
        print(f"Missing expected attribute: {e}")
        return []

    except Exception as e:
        print(f"An unexcepted error occured: {e}")
        return []

# Scrape all books under category across webpages
def scrape_category(name, url):
    """Scrape all books in a category, following pagination"""
    # To store information about the book found
    books = []
    # Start the first page of the category
    next_url = url

    try:
        # page should contains a valid url
        while next_url:
            soup = get_text(next_url)

            if soup is None:
                print(f"Failed to fetch page: {next_url}")
                break

            # Select every book on the page
            articles = soup.select("article.product_pod")

            # Extract book details like title, price, ratings, stock availability
            for article in articles:
                try:
                    # Get book title
                    title = article.h3.a["title"]
                    # Get book price
                    price = article.select_one("p.price_color").text.strip()
                    # Get rating
                    star_rating = article.p["class"][1]
                    # Get instock availability
                    availability = article.select_one("p.instock.availability").text.strip()

                    # create a book dictionary to store the book's information
                    books.append({
                        "title": title,
                        "price": price,
                        "star_rating": star_rating,
                        "availability": availability,
                        "category": name
                    })

                except (AttributeError, KeyError, TypeError) as e:
                    print(f"Skipping a book because of missing data: {e}")
                    continue

            # Check if there is another page (ie., if it contains <li class="next"> with <a> tag)
            next_page = soup.select_one("li.next a")

            if next_page:
                next_href = next_page["href"]
                # Split the next_url from the right & only once.
                parts = next_url.rsplit("/", 1)
                # Get the first part [0]
                base_path = parts[0]
                # add slash ("/") and append next_page
                next_url = base_path + "/" + next_href
            else:
                # If there is no next page then loop ends
                next_url = None

            # Pause before the next request
            time.sleep(0.5)

    except Exception as e:
        print(f"Error while scraping category '{name}': {e}")

    return books

# Gather books from multiple categories & combine them into a Pandas DataFrame
def scrape_books(n_categories=5):
    """Scrape books from the first n_categories and return a DataFrame."""

    # To store every book from every category
    all_books = []

    try:
        # Get the category names and its url
        categories = get_category_links(n_categories)

        # Check whether categories were found
        if not categories:
            print("No categories found.")
            # returns an empty DataFrame
            return pd.DataFrame()

        for name, url in categories:
            try:
                # Print current category
                print(f"Scraping Category: {name}")

                # Scrape all the books in that category
                books = scrape_category(name, url)
                # append these books to the master list
                all_books.extend(books)

            except Exception as e:
                print(f"Error scraping category '{name}': {e}")
                continue

        # Convert list into a DataFrame
        return pd.DataFrame(all_books)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

# Run the scraper
try:
    df = scrape_books(n_categories=5)

    # Check if the DataFrame is empty
    if df.empty:
        print("No books were scraped.")
    else:
        print(f"Total books scraped: {len(df)}")
        print(f"Categories covered: {df['category'].nunique()}")

        # Display the first 5 rows
        #print(df.head())

except KeyError as e:
    print(f"Missing expected column: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Fixed exchange rate (GBP to INR) as on 02-Aug-2026
fixed_rate_gbp_to_inr = 128.61

# To clean the price column
def clean_price(price_str):
    """Strip GBP symbol and convert to float. Return NaN if conversion fails"""
    try:
        # remove "Â£" symbol and extra spaces 
        cleaned_price = price_str.replace("Â£", "").strip()
        # convert price to float
        price = float(cleaned_price)
        return price
    
    except ValueError as e:
        print(f"Error in Value: {e}")
        return np.nan
    except AttributeError as e:
        print(f"Missing attribute: {e}")
        return np.nan

# To clean the rating column
def clean_rating(rating_str):
    """Convert text rating (one - Five) to integer (1-5). Returns NaN if unrecognized"""
    try:
        # mapping the rating (str) values to rating (int) values
        rating_map = {"One": 1,
                      "Two": 2,
                      "Three": 3,
                      "Four": 4,
                      "Five": 5}
        # if rating is not in (one - five) then np.nan - missing value
        rating = rating_map.get(rating_str, np.nan)
        return rating

    except Exception as e:
        print(f"Error while cleaning the rating: {e}")
        return np.nan

# To clean availability column
def clean_availability(avail_str):
    """Parse availability text into boolean. Return None if unparseable"""
    try:
        # Check whether the input is a string
        if not isinstance(avail_str, str):
            return None

        # Convert the string to lowercase
        text = avail_str.lower()

        # Checks whether the given phrase or input string exists
        if "in stock" in text:
            return True
        elif "out of stock" in text:
            return False
        else:
            return None
    
    except Exception as e:
        print(f"Error while cleaning the availability: {e}")
        return None

# Apply the above data clean functions
try:
    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)
    df["in_stock"] = df["availability"].apply(clean_availability)

except KeyError as e:
    print(f"Missing column in DataFrame: {e}")

except Exception as e:
    print(f"Unexpected error while cleaning columns: {e}")

# Handling missing values
try:
    # Count no. of missing rows in the price & rating columns
    n_price_missing = df["price_gbp"].isna().sum()
    n_rating_missing = df["rating"].isna().sum()

    # For numeric fields - median imputation
    price_median = df["price_gbp"].median()
    rating_median = df["rating"].median()

    # Replace missing values with median in price & rating columns and convert rating column to integers 
    df["price_gbp"] = df["price_gbp"].fillna(price_median)
    df["rating"] = df["rating"].fillna(rating_median).astype(int)

except Exception as e:
    print(f"Error while handling missing values: {e}")

# Drop rows with invalid availability
try:
    # No. of rows before dropping
    n_before = len(df)
    # Select rows where in_stock is not missing and create an independent copy of the filtered Dataframe
    df = df[df["in_stock"].notna()].copy()
    # No. of rows dropped
    n_dropped = n_before - len(df)
    # Convert the data type of the column in_stock to Boolean
    df["in_stock"] = df["in_stock"].astype(bool)

except Exception as e:
    print(f"Error while parsing the availability column: {e}")

# Print Summary 
try:
    print(f"Price rows imputed: {n_price_missing}")
    print(f"Rating rows imputed: {n_rating_missing}")
    print(f"Rows dropped due to unparseable availability: {n_dropped}")

except NameError as e:
    print(f"Summary variables are not available because an earlier step failed: {e}")

except Exception as e:
    print(f"Error while printing the summary: {e}")

# Currency conversion
try:
    df["price_inr"] = df["price_gbp"] * fixed_rate_gbp_to_inr

except Exception as e:
    print(f"Error while converting GBP to INR: {e}")

# Print cleaned data
try:
    print(df.head())

except Exception as e:
    print(f"Error while printing the DataFrame: {e}")
