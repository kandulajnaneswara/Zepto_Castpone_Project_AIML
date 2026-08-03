Data Pipeline Module

This module (/data_pipeline) covers the scraping, cleaning, and database-loading stages of the Zepto capstone project. It scrapes book data from books.toscrape.com, cleans and type-converts the fields, loads the result into a normalized SQLite database, and runs a set of required SQL queries against it.

Contents
data_pipeline.py — single script containing the full pipeline: scraping → cleaning → currency conversion → SQLite schema creation → data insertion → SQL queries → pandas verification.
books_toscrape.db — the SQLite database produced by running the script.
output_file.csv — intermediate CSV of the raw scraped + cleaned data (saved before database loading).
Install / Run Steps
Install dependencies (listed in the root requirements.txt):
bash
   pip install -r requirements.txt

Run the full pipeline:
bash
   python data_pipeline.py

This single script performs, in order:

Scraping all books across the first 10 categories on books.toscrape.com, following pagination within each category
Saving raw scraped data to output_file.csv
Cleaning/type-converting fields and re-saving the cleaned data
Creating the SQLite schema (categories, books tables) in books_toscrape.db
Inserting the cleaned data into the database
Running 6 SQL queries (SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, JOIN) and printing their output
Running an additional JOIN query (top 10 rated books per category)
Reading 2 query results back with pd.read_sql and reproducing the JOIN result independently with pd.merge, then comparing both for same results.

All output (scrape progress, no. of rows, query results, and the SQL-vs-pandas comparision check) prints directly to the vscode output console.

Database Schema

Two tables, related by a primary/foreign key:

sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL,
    price_inr   REAL,
    rating      INTEGER,
    in_stock    INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
Design Decisions made
Scraping scope: first 10 categories, following "next page" pagination within each, to ensure comfortably more than the minimum required rows and categories.
Field cleaning:
price → stripped of the currency symbol and converted to price_gbp (float).
star_rating text (One…Five) → mapped to an integer rating (1–5).
availability text → parsed into a boolean in_stock.
Handling unparseable rows:
price_gbp and rating (numeric fields) → missing/unparseable values are median-imputed.
in_stock (boolean field) → rows with unparseable availability text are dropped.
Currency conversion: price_inr is computed using a fixed, project-defined baseline rate of 1 GBP = 105.50 INR. This is an artificial constant specified by the assignment, not a live or historical market rate — it requires no API call, no network access, and no date reference.
Database regeneration: the script creates the schema with DROP TABLE IF EXISTS before creating fresh tables, so re-running data_pipeline.py from scratch always regenerates an identical database from the scraped source data — no manual database setup is required.
Required SQL Queries

The script executes and prints output for the following, collectively covering every required clause:

SELECT / WHERE — in-stock books
ORDER BY — books sorted by rating (descending)
LIMIT — top 5 most expensive books (by INR price)
DISTINCT — distinct category IDs present in the books table
BETWEEN — books priced between 10 and 30 GBP
JOIN — books joined with their category names, ordered by category then rating

An additional JOIN query (top 10 rated books per category, using ROW_NUMBER() OVER (PARTITION BY ...)) is included for "top N per category" requirement.

SQL vs. Pandas Verification

Two of the query results above are read back into DataFrames using pd.read_sql(...). The JOIN query's result is separately reproduced using pd.merge(...) directly on the categories and books DataFrames (no SQL). Both are sorted identically and compared with .equals(...) to confirm they produce same outputs — the script prints True/False for this check in the console.