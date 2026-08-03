Install the below libraries before executing the .py file

1. pip install beautifulsoup4
2. pip install requests
3. pip install pandas
4. pip install numpy
5. pip install sqlite3


Work flow:
1. Assign the website's url address to "base_url" variable
2. create a function "get_text" for a response object to get the web page's HTML content
	a. Use try and except calls if any failure occurs while accessing the webpage
	b. check the status of the response and raise an exception for HTTP errors
	c. create beautiful soup object to parse HTML text with the help of html.parser
	d. print the exceptions of the requested url as below:
		HTTPError - if webpage status is 404 (Not Found) or 500 (Internal Server Error)
		ConnectionError - if unable to connect to the server
		Timeout - if server took too long to respond
		RequestException - if any other request related error
3. create a function "get_category_links" to find all categories listed on the webpage
	a. the function should return the first 5 categories by default
	b. get the webpage address, download the homepage, parse the text with beautifulsoup, stores the parsed HTML (in soup) from the function "get_text"
	c. check if the webpage was downloaded else it should return an empty list.
	d. search for every category link inside the sidebar (the sidebar contains all <a> tags)
	e. create an empty list to store the results
	f. loop the first 5 category links to extract the category_name, url(href attribute), create the full url (base_url + href) and append the information to the result list.
	g. if there is no response from the webpage it crashes. So, to prevent the crashing we use exceptions
		AtrributeError - if any expected element isn't found or doesn't contain expected HTML structure
		KeyError - if an expected attribute like href is missing from <a> tag
		Exception - if any unexpected error occurred.
4. create a function "scrape_category" to scrape all books under category across pages
	a. fetch all books listed on the current page
	b. build a dictionary to store the extracted data
	c. append the dictionary to a list
	d. go to the next page if current page is not the last one
		we use rsplit() because it splits from the right. split() uses left split by default
	e. pause the scraper 0.5 sec before requesting the next page. 
		This reduces the load on the website
5. create a function "scrape_books" to gather book's information from multiple categories and combine them into a single Pandas DataFrame.
	a. create an empty list to store every book from every category
	b. get category names and links from the previous functions
	c. check whether categories were found
		if the list is empty, an empty DataFrame will be returned
	d. loop through each category to get the category_name and url
	e. print the current category
	f. scrape all books in that category by using the previous function
	g. add these books to the created empty list at the start of the function
	h. convert the list into a Panda's DataFrame
6. Save dataframe to excel file(.csv) 
7. Try to run the workflow upto the dataframe
	a. load the dataframe in "df" from the excel file
	b. if df is empty, then print "no books were scraped"
	c. else, print the total books scraped and no. of categories covered from the scrapped books

8. Fixed exchange rate
	a. define fixed exchange rate
9. EDA
	a. clean price column (remove currency symbol, extra spaces and convert to float)
	b. clean rating column (map the string ratings to numerical ratings)
	c. clean availability column (valid value is (in stock & out of stock), else for other values its None)
	d. apply cleaning functions discussed above on the original columns by creating independent copy of the dataframe
	e. Handle missing values in the columns
	f. Print the summary of the cleaning columns
	g. convert currency and print the final dataframe
10. SQLite3
	a. create a file .db with a database path
	b. create a normalized SQLite schema function
	c. establish the connection to the database using ".connect()"
	d. create a cursor object using ".cursor()"
	e. execute multiple SQL statements with one go using ".executescript()". If one SQL statement execution, use ".execute()"
	f. save changes using ".commit()"
	g. create an exception to Undo changes if any error occurs to the database using ".rollback()"
 
