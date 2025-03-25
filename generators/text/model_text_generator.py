#!/usr/bin/env python3

import logging
import threading
import time
import random

from pathlib import Path
from configparser import ConfigParser

from simpletransformers.language_generation import LanguageGenerationModel
from llama_cpp import Llama

from reddit_io.tagging_mixin import TaggingMixin
from bot_db.db import Thing as db_Thing

from utils.keyword_helper import KeywordHelper
from utils.toxicity_helper import ToxicityHelper

from utils.memory import get_available_memory
from utils import ROOT_DIR


class ModelTextGenerator(threading.Thread, TaggingMixin):

	daemon = True
	name = "MTGThread"

	# GPU is not really required for text generation
	# So the default is False
	_use_gpu = False

	_config = None

	# The amount of memory required to start generation, in KB
	# This is the default for GPT-2 Small (117M parameters)
	# This will need to be increased for larger GPT-2 models
	_memory_required = 1400000

	def __init__(self, username, temp=0.8):
		threading.Thread.__init__(self)

		self._config = ConfigParser()
		self._config.read('ssi-bot.ini')
		self.llama = None
		self.subreplace = None
		
		self.username = username
		self.name = f"{username}_MTG"
		self.temperature = float(temp)
		if self._config[self.username]["text_model_path"].endswith("gguf"):
			self.llama = Llama(self._config[self.username]["text_model_path"], use_mmap=True, use_mlock=True, n_ctx=4096, n_batch=1024, n_threads=6, n_threads_batch=12)
			self.logit_bias = {self.llama.token_eos: -10.0}

		if "subreplace" in self._config[self.username]:
			self.subreplace = self._config[self.username]["subreplace"].split(",")

		if "end_token" in self._config[self.username]:
			self._end_tag = self._config[self.username]["end_token"]

		

		# Configure the keyword helper to check negative keywords in the generated text
		self._toxicity_helper = ToxicityHelper()

	def run(self):

		logging.info("Starting GPT-2 text generator daemon")

		while True:

			jobs = self.top_pending_jobs()

			if not jobs:
				# there are no jobs at all in the queue
				# Rest a little before attempting again
				time.sleep(30)
				continue

			if get_available_memory(self._use_gpu) < self._memory_required:
				# Not enough memory.. Sleep and start again
				logging.info('Insufficient memory to generate text')
				time.sleep(30)
				continue

			for job in jobs:

				try:
					logging.info(f"Starting to generate text for bot {job.bot_username}, job_id {job.id}.")

					# use the model to generate the text
					# pass a copy of the parameters to keep the job values intact

					if self.subreplace is not None and job.subreddit is not None:
						newsub = random.choice(self.subreplace)
						while job.subreddit in job.text_generation_parameters["prompt"]:
							job.text_generation_parameters["prompt"] = job.text_generation_parameters["prompt"].replace(job.subreddit, newsub)

					generated_text = self.generate_text(job.bot_username, job.text_generation_parameters.copy())

					if generated_text:
						####added by Krobix
						irp = ["[removed]", "[deleted]"]
						c=0
						tries=0
						max_tries=10
						while c<len(irp):
							if tries>=max_tries:
								break

							if type(generated_text) is tuple:
								gen = generated_text[1]
							else:
								gen = generated_text
							
							if irp[c] in gen:
								c=0
								generated_text = self.generate_text(job.bot_username, job.text_generation_parameters.copy())
								tries+=1
								continue
							else:
								c+=1
							
						if tries>=max_tries:
							continue

						if "nofilter" in job.text_generation_parameters:
							no_filter = job.text_generation_parameters["nofilter"]
						else:
							no_filter = False

						if type(generated_text) is tuple:
							generated_text = "".join(generated_text)
						###########

						# Check for any negative keywords in the generated text and if so, return nothing
						negative_keyword_matches = self.test_text_against_keywords(job.bot_username, generated_text)
						if negative_keyword_matches and (not no_filter):
							# A negative keyword was found, so don't post this text back to reddit
							logging.info(f"Negative keywords {negative_keyword_matches} found in generated text, this text will be rejected.")
							continue

						# Perform a very basic validation of the generated text
						prompt = job.text_generation_parameters.get('prompt', '')
						valid = self.validate_generated_text(job.source_name, prompt, generated_text)
						if not valid:
							logging.info(f"Generated text for {job} failed validation, this text will be rejected.")
							continue

						toxicity_failure = self.validate_toxicity(job.bot_username, prompt, generated_text)
						if toxicity_failure and (not no_filter):
							logging.info(f"Generated text for {job} failed toxicity test, this text will be rejected.-> {generated_text}")
							continue

						# if the model generated text, set it into the 'job'
						job.generated_text = generated_text.replace('<|n|>','\n')
						job.save()

				except:
					logging.exception(f"Generating text for job {job} failed")

				finally:
					# Increment the counter because we're about to generate text
					job.text_generation_attempts += 1
					job.save()

	def generate_text(self, bot_username, text_generation_parameters):

		model_path = self._config[bot_username]['text_model_path']
		prompt = text_generation_parameters.pop('prompt', '')

		if self.llama is not None:
			logging.info("Generating text using llama")
			
			gen = self.llama(prompt=prompt, temperature=float(self.temperature), max_tokens=512, logit_bias=self.logit_bias)["choices"][0]["text"]
			#gen += self._end_tag
			logging.info(f"llama finished generating: {str(gen)}")
			#llama is too fucking fast apparently?
			#time.sleep(120)
			return (prompt,str(gen))

		# if you are generating on CPU, keep use_cuda and fp16 both false.
		# If you have a nvidia GPU you may enable these features
		# TODO shift these parameters into the ssi-bot.ini file
		model = LanguageGenerationModel("gpt2", model_path, use_cuda=self._use_gpu, args={'fp16': False})

		start_time = time.time()

		# pop the prompt out from the args
		#set temp
		text_generation_parameters["temperature"] = float(self.temperature)
		
		#if len(prompt)>2048:
		#	prompt = prompt[len(prompt)-2048:]#b
		#	promptl = prompt.split(" ")
		#	if not prompt.startswith("<|"):
		#		promptl.pop(0)
		#	prompt = " ".join(promptl)

		output_list = model.generate(prompt=prompt, args=text_generation_parameters)

		end_time = time.time()
		duration = round(end_time - start_time, 1)

		logging.info(f'{len(output_list)} sample(s) of text generated in {duration} seconds.')

		if output_list:
			return output_list[0]

	def top_pending_jobs(self):
		"""
		Get a list of jobs that need text to be generated, by treating
		each database Thing record as a 'job'.
		Three attempts at text generation are allowed.

		"""
		query = db_Thing.select(db_Thing).\
					where(db_Thing.status == 3).\
					where(db_Thing.bot_username == self.username). \
					order_by(db_Thing.created_utc)
		return list(query)

	def test_text_against_keywords(self, bot_username, generated_text):
		# Load the keyword helper with this bot's config
		keyword_helper = KeywordHelper(bot_username)
		return keyword_helper.negative_keyword_matches(generated_text)

	def validate_toxicity(self, bot_username, prompt, generated_text):

		# Remove tags from the
		new_text = generated_text[len(prompt):]
		tagless_new_text = self.remove_tags_from_string(new_text)

		# Reconfigure the toxicity helper to use the bot's config
		self._toxicity_helper.load_config_section(bot_username)
		return self._toxicity_helper.text_above_toxicity_threshold(tagless_new_text)

	def validate_generated_text(self, source_name, prompt, generated_text):

		if source_name == 't3_new_submission':
			# The job is to create a new submission so
			# Check it has a title
			title = self.extract_title_from_generated_text(generated_text)
			if title is None:
				logging.info("Validation failed, no title")
			return title is not None

		else:
			# The job is to create a reply
			# Check that is has a closing tag
			new_text = generated_text[len(prompt):]
			if not self._end_tag in new_text:
				logging.info("Validation failed, no end tag")
			return self._end_tag in new_text
