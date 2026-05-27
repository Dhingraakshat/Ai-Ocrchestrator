import os
import json
import schedule
import time
from datetime import datetime
from utils import with_retry
from llm import call_groq
import logging
from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger('jarvis')

def get_post_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load post data: {e}")
        return None

def publish_post(post_data):
    try:
        linkedin_api_url = os.getenv('LINKEDIN_API_URL')
        linkedin_api_token = os.getenv('LINKEDIN_API_TOKEN')
        headers = {
            'Authorization': f'Bearer {linkedin_api_token}',
            'Content-Type': 'application/json'
        }
        response = requests.post(linkedin_api_url, headers=headers, json=post_data)
        if response.status_code == 201:
            logger.info("Post published successfully")
            return True
        else:
            logger.error(f"Failed to publish post: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to publish post: {e}")
        return False

def schedule_post(post_data):
    try:
        scheduling_time = post_data['scheduling_time']
        schedule.every().day.at(scheduling_time).do(publish_post, post_data)
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to schedule post: {e}")

@with_retry
def run_LinkedInDailyPostAgent(query=''):
    try:
        post_data_file_path = 'post_data.json'
        post_data = get_post_data(post_data_file_path)
        if post_data:
            if 'scheduling_time' in post_data:
                schedule_post(post_data)
            else:
                publish_post(post_data)
        else:
            logger.error("No post data found")
    except Exception as e:
        logger.error(f"Failed to run LinkedInDailyPostAgent: {e}")