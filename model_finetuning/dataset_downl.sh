#!/bin/bash
until timeout --foreground 1h ~/venv/bin/python3 download_reddit_finetuning_data.py; do
	echo Exception occured or timed out to avoid error, continuing in 20 mins
	sleep 1200
done 
