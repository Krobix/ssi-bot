#!/usr/bin/env python3

import json
import logging
import nltk
#import requests
import threading
import time
import urllib.parse

from collections import OrderedDict

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk import pos_tag, TweetTokenizer

from reddit_io.tagging_mixin import TaggingMixin

from bot_db.db import Thing as db_Thing

from duckduckgo_images_api import search
import random

class ImageScraper(threading.Thread, TaggingMixin):

	daemon = True
	name = "ImageScraper"

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):

		while True:

			# get the top job in the list
			jobs = self.top_pending_jobs()

			for job in jobs:

				try:
					logging.info(f"Starting to find an image for job_id {job.id}.")

					if not job.image_generation_parameters['prompt'] and job.generated_text:
						# If there is no prompt, but is generated text, attempt to extract the title
						# from the generated text and use it as the prompt
						job.image_generation_parameters['prompt'] = self.extract_title_from_generated_text(job.generated_text)

					image_url = self._download_image_for_search_string(job.bot_username, job.image_generation_parameters.copy(), job.image_generation_attempts)

					if image_url:
						logging.info(f'Using image url for job {job}: {image_url}')
						job.generated_image_path = image_url
						job.save()

					# Sleep a bit here to not hammer the servers
					time.sleep(10)

				except:
					logging.exception(f"Scraping image for a {job} failed")

				finally:
					job.image_generation_attempts += 1
					job.save()

			# Sleep a bit more to be nice to dem servers
			time.sleep(120)

	def _download_image_for_search_string(self, bot_username, image_generation_parameters, attempt):

		logging.info(f"{bot_username} is searching on DuckDuckGo for an image..")

		# pop the prefix out from the parameters
		search_prefix = image_generation_parameters.pop('image_post_search_prefix', None)

		# Split the search prefix into keywords
		search_prefix_keywords = search_prefix.split(' ') if search_prefix else []

		prompt = image_generation_parameters.pop('prompt', None)
		prompt_keywords = []

		if prompt:
			first_sentence = sent_tokenize(prompt)[0]

			# remove numbers and tokenize the text
			tokenized = TweetTokenizer().tokenize(first_sentence.translate({ord(ch): None for ch in '0123456789'}))
			# remove single letter tokens
			tokenized = [i for i in tokenized if len(i) > 1]
			# remove duplicates from the token list
			tokenized = list(OrderedDict.fromkeys(tokenized))

			# put nltk tags on it
			pos_tagged_text = nltk.pos_tag(tokenized)

			# Extract all nouns, verbs and adverbs and append to the existing
			prompt_keywords = [i[0] for i in pos_tagged_text if i[1][:2] in ['NN', 'VB', 'RB']]

		# Merge the prefix keywords and prompt keywords, to a maximum of 10.
		search_keywords = search_prefix_keywords + prompt_keywords[:(10 - len(search_prefix_keywords))]

		# Convert all of the keywords back into a single string
		search_keywords_as_string = ' '.join(search_keywords)

		results = search(search_keywords_as_string)

		r = random.randint(0,len(results)-1)
		image_url = r["image"]
		return image_url

	def top_pending_jobs(self):
		"""
		Get a list of jobs that need an image to be found via the scraper

		"""

		query = db_Thing.select(db_Thing).\
					where(db_Thing.image_generation_parameters['type'] == 'scraper').\
					where(db_Thing.status == 5).\
					order_by(db_Thing.created_utc)
		return list(query)
