# mboxEmailAnalyser

This project was created to analyze email from the Decrypteur at Radio-Canada. The project took place during the summer of 2020 as an internship with IVADO "des données pour raconter" in data journalism. The goal is to understand the theme of the questions sent to the Decrypteur. Emails are generally formed as a question asking about the information in certain URL or about a specific topic found on social media platforms such as Facebook. 

The code allows parsing of mobx files from Google takeout to csv files. It extracts data such as TO, FROM, DATE, BODY, SUBJECT, ATTACHMENT, ATTACHMENT_TYPE. This functionality is built in the EmailParser dans MboxParser.

Furthermore, specific analytic tools where build to clean and search the data for analysis (EmailDataFrame), add more data from Facebook URLs (FacebookScraper), YouTube URLs (YoutubeSraper). 


- analyser_tools : Data structure and functions used in the notebooks
- data : Data for input in the analysis (.mbox, theme_words.csv)
- output : Graph, csv, logs, pickles (not everything is on the github for privacy concerns)
- notebook : final version of the code used for the article with Radio-Canada

Article link
https://ici.radio-canada.ca/info/2020/10/pandemie-decrypteurs-courriels-complot-covid-19-coronavirus-fausses-nouvelles/?fbclid=IwAR0PQZd9Y_q0ZYH6ylJp0ieDvI_dZnMqaIoevYZD9hOlFtG2s0dkBvuGDMk
