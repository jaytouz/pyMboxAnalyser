# mboxEmailAnalyser

This project was created to analyse email from the Decrypteur at Radio-Canada. The projet took place during the summer of 2020 as an internship with IVADO "des données pour raconter" in data journalism. The goal is to understand the theme of the question sent to the Decrypteur. Email are generaly formed as a question asking about the information in certain url or about a specific topic found on social media plateform such as Facebook. 

The code allows to parse mbox file from google takeout to csv file. It extract data such as TO, FROM, DATE, BODY, SUBJECT, ATTACHMENT, ATTACHMENT_TYPE. This functionality is build in the EmailParser dans MboxParser.

Furthermore, specific analytic tools where build to clean and search the data for analysis (EmailDataFrame), add more data from Facebook urls (FacebookScraper), Youtube urls (YoutubeSraper). 


- analyser_tools : Data structure and functions used in the notebooks
- data : Data for input in the analysis (.mbox, theme_words.csv)
- output : Graph, csv, log, pickle (not everything is on the github for privacy concerns)
- notebook : final version of the code used for article with Radio-Canada
