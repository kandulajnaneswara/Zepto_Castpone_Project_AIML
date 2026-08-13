import requests
import pandas as pd
import numpy as np
import sqlite3
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
def get_category_links(n_categories = 10):
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
def scrape_books(n_categories=10):
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

# Save the DataFrame to excel.
try:
    csv = scrape_books(n_categories= 10).to_csv("output_file.csv", index = False)
    print("File saved successfully")

except PermissionError as e:
    print(f"Error: Permission denied {e}. Close the file if it's open in excel.")

except FileNotFoundError as e:
    print(f"Error: {e}. The specified folder directory does not exist.")

except Exception as e:
    print(f"An Unexpected error occurred: {e}")

# Run the scraper
file_path = "output_file.csv"
try:
    try:
        # Load the saved csv file 
        df = pd.read_csv(file_path)
        print("File loaded successfully")
        # Check if the DataFrame is empty
        if df.empty:
            print("No books were scraped.")
        else:
            print(f"Total books scraped: {len(df)}")
            print(f"Categories covered: {df['category'].nunique()}")

            # Display the first 5 rows
            #print(df.head())

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Check the file path.")

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is blank or contains no data.")

    except pd.errors.ParserError:
        print("Error: The file is corrupted or poorly formatted (e.g., mismatched columns).")

    except PermissionError:
        print("Error: Permission denied. Close the file if it is open in another program.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

except KeyError as e:
    print(f"Missing expected column: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Fixed exchange rate (GBP to INR) - defined in the requirement
fixed_rate_gbp_to_inr = 105.50

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
    df["price_inr"] = (df["price_gbp"] * fixed_rate_gbp_to_inr).round(2)

except Exception as e:
    print(f"Error while converting GBP to INR: {e}")

# Print cleaned data
try:
    csv = df.to_csv("Final_output_file.csv", index = False)
    print("File saved successfully")
    print(df.head())

except PermissionError:
    print("Error: Permission denied. Close the file if it is open in Excel.")

except FileNotFoundError:
    print("Error: The specified folder directory does not exist.")

except Exception as e:
    print(f"Error while printing the DataFrame: {e}")

# Database Path
db_path = "books_toscrape.db"

# Create a normalized SQLite schema 
def create_schema(conn):
    """Create a normailized database schema by dropping existing tables &
       creating fresh categories and books tables."""

    try:
        # Connect to the database (creates file if it doesn't exist)
        #conn = sqlite3.connect(db_path)
        # create a cursor object
        cur = conn.cursor()

        # Execute multiple SQL statements 
        cur.executescript("""
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;
        
        CREATE TABLE categories (category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                 category_name TEXT UNIQUE NOT NULL);
                                 
        CREATE TABLE books (book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title TEXT NOT NULL,
                            price_gbp REAL,
                            price_inr REAL,
                            rating INTEGER,
                            in_stock INTEGER,
                            category_id INTEGER,
                            FOREIGN KEY (category_id) REFERENCES categories(category_id));
                            """)
        # Save changes permanently
        conn.commit()

        print(f"Database schema created successfully")

    except sqlite3.Error as e:
        # Undo changes if any error occurs
        conn.rollback()
        print(f"SQLite Error: {e}")

    except Exception as e:
        # Handles any other unexpected error
        print(f"Unexpected Error: {e}")

# Insert data from Pandas dataframe into SQLite database.
def populate_tables(conn, df):
    """Populates the categories and books tables using data from the DataFrame"""

    try:
        # create cursor()
        cur = conn.cursor()

        # Insert unique categories
        categories = df["category"].unique()

        for cat in categories:
            # Insert one category into one database
            cur.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))

        # save inserted categories
        conn.commit()

        # Build category lookup dictionary
        cur.execute("SELECT category_id, category_name FROM categories")

        # create cat_lookup dictionary
        cat_lookup = {}

        # Fetch all rows from the database
        rows = cur.fetchall()

        # Loop through each row (store name as the key and cid as value)
        for cid, name in rows:
            cat_lookup[name] = cid

        # Prepare book records
        # create an empty list
        rows = []

        # Loop through each row in the dataframe
        for _, row in df.iterrows():
            # Get the required values from the current row
            title = row["title"]
            price_gbp = row["price_gbp"]
            price_inr = row["price_inr"]
            rating = int(row["rating"])
            in_stock = int(row["in_stock"])
            category_id = cat_lookup[row["category"]]

            # create tuple with the above values
            data = (title,
                    price_gbp,
                    price_inr,
                    rating,
                    in_stock,
                    category_id,)
            
            # Append the tuple to the list
            rows.append(data)

        # Insert books (insert all rows together)
        cur.executemany("""INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
                        VALUES (?, ?, ?, ?, ?, ?)""", rows)

        # save all books 
        conn.commit()
        print("Data inserted successfully")

    except KeyError as e:
        conn.rollback()
        print(f"Missing DataFrame column: {e}")

    except sqlite3.Error as e:
        conn.rollback()
        print(f"SQLite Error: {e}")

    except Exception as e:
        conn.rollback()
        print(f"Unexpected Error: {e}")

# SQL program execution
try:
    # Connect to the database (creates file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    populate_tables(conn, df)

except sqlite3.Error as e:
    print(f"Database Connection Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")


# SQL Queries execution
def run_query(conn, label, query, paramas = ()):
    """Executes an SQL Query and Print the results"""

    try:
        print(f"\n-----   {label}   -----")
        print(query)

        # Create cursor
        cur = conn.cursor()

        # Execute SQL query
        cur.execute(query, paramas)

        # Fetch all records
        results = cur.fetchall()

        # Print each row
        for row in results:
            print(row)

        return results

    except sqlite3.Error as e:
        print(f"SQLite Error while executing '{label}': {e}")
        return []

    except Exception as e:
        print(f"Unexpected Error while executing '{label}': {e}")
        return []

# Execute queries
try:
    # SELECT / WHERE
    q1 = """SELECT title, price_gbp, rating
            FROM books
            WHERE in_stock = 1;"""

    run_query(conn, "Q1: SELECT --> in_stock books", q1)

    # ORDER BY
    q2 = """SELECT title, rating
            FROM books
            ORDER BY rating DESC;"""

    run_query(conn, "Q2: ORDER BY --> Books by rating descending order", q2)

    # 3. LIMIT
    q3 = """SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 5;"""

    run_query(conn, "Q3: LIMIT --> Top 5 most expensive books", q3)

    # 4. DISTINCT
    q4 = """SELECT DISTINCT category_id
            FROM books;"""

    run_query(conn, "Q4: DISTINCT --> Unique category IDs", q4)

    # 5. BETWEEN
    q5 = """SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30;"""

    run_query(conn, "Q5: BETWEEN --> Books priced between 10 and 30 GBP", q5)

    # 6. JOIN
    join_query = """SELECT c.category_name, b.title, b.rating
                    FROM books b
                    JOIN categories c ON b.category_id = c.category_id
                    ORDER BY c.category_name, b.rating DESC;"""

    join_results = run_query(conn, "Q6: JOIN --> Books with Category Names", join_query)

except Exception as e:
    print(f"Error while executing queries: {e}")

# 10 highest rated books per category
try:
    q_top10_per_category = """SELECT category_name, title, rating 
                              FROM (SELECT c.category_name AS category_name, b.title AS title, b.rating AS rating,
                                    ROW_NUMBER() OVER (PARTITION BY c.category_id ORDER BY b.rating DESC) AS rn
                                    FROM books b
                                    JOIN categories c ON b.category_id = c.category_id)
                              WHERE rn <= 10;"""

    # Execute the query  
    top_10_results = run_query(conn, "JOIN: top 10 rated books per category", q_top10_per_category)

except sqlite3.Error as e:
    print(f"SQLite Error while executing Top 10 query: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")


# Read two query results back into DataFrame (pd.read_sql, pd.merge)
try:
    # Read query 1 results into DataFrame
    df_instock = pd.read_sql(q1, conn)

    print("Read query 1 back via pd.read_sql (Q1):")
    print(df_instock.head())

    # Read JOIN query result into DataFrame
    df_join_sql = pd.read_sql(join_query, conn)

    print("\nRead JOIN query back via pd.read_sql (JOIN Query):")
    print(df_join_sql.head())

    # Read complete categories & books tables
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    books_df = pd.read_sql("SELECT * FROM books", conn)

    # Perform join using Pandas library
    df_join_pandas = (books_df.merge(categories_df, on = "category_id", how = "inner")
                      [["category_name", "title", "rating"]].sort_values(["category_name", "rating"],
                                                                         ascending= [True, False]).reset_index(drop= True))

    # Sort SQL join output
    df_join_sql_sorted = (df_join_sql.sort_values(["category_name", "rating"], ascending= [True, False]).reset_index(drop= True))

    print("\nSQL JOIN RESULT\n")
    print(df_join_sql_sorted.head())

    print("\nPANDAS MERGE RESULT\n")
    print(df_join_pandas.head())

    # Compare both DataFrames
    compare_df = df_join_sql_sorted.equals(df_join_pandas)

    print(f"\nSQL JOIN and PANDAS MERGE produce equivalent output: {compare_df}")

except pd.errors.DatabaseError as e:
    print(f"Pandas Database Error: {e}")

except sqlite3.Error as e:
    print(f"SQLite Error: {e}")

except KeyError as e:
    print(f"Missing Column Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")

finally:
    # Close the database connection now that all queries/comparisions are done
    if "conn" in locals():
        conn.close()
        print("Database connection closed")