from dotenv import load_dotenv
import praw
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import cv2

load_dotenv()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(dir_path, "images/")
ignore_path = os.path.join(dir_path, "ignore_images/")

create_folder(image_path)
create_folder(ignore_path)

reddit = praw.Reddit(
    client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    user_agent=os.getenv('user_agent'),
    username=os.getenv('user'),
    password=os.getenv('password')
)

data = []

ignore_images = []
for (dirpath, _, filenames) in os.walk(ignore_path):
    for file in filenames:
        img_path = os.path.join(dirpath, file)
        img = cv2.imread(img_path)
        if img is not None:
            ignore_images.append(cv2.resize(img, (224, 224)))

time_threshold = datetime.now() - timedelta(days=30)

with open("sub_list.csv","r") as f:
    for line in f:
        sub = line.strip()
        subreddit = reddit.subreddit(sub)
        print(f"Starting {sub}")
        count = 0

        for submission in subreddit.new(limit=5000):
            post_time = datetime.utcfromtimestamp(submission.created_utc)
            if post_time < time_threshold:
                continue
            d = {
                'title': submission.title,
                'text':submission.selftext,
                'created':datetime.fromtimestamp(submission.created_utc),
                'upvotes':submission.score,
                'flair':submission.link_flair_text,
                'image':None,
            }
            url = submission.url.lower()
            if "jpg" in url or "png" in url:
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code != 200:
                        print(f"Failed with status {resp.status_code}: {url}")
                        continue

                    image_data = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                    if image is None:
                        print(f"OpenCV decode failed for: {url}")
                        continue

                    compare_image = cv2.resize(image, (224, 224))

                    
                    ignore_flag = False
                    for ignore in ignore_images:
                        difference = cv2.subtract(ignore, compare_image)
                        b, g, r = cv2.split(difference)
                        total_diff = (
                            cv2.countNonZero(b) +
                            cv2.countNonZero(g) +
                            cv2.countNonZero(r)
                        )
                        if total_diff == 0:
                            ignore_flag = True
                            break

                    if not ignore_flag:
                        out_name = f"{sub}-{submission.id}.png"
                        cv2.imwrite(os.path.join(image_path, out_name), image)
                        count += 1
                        d['image'] = os.path.join(image_path, out_name)

                except Exception as e:
                    print(f"Image failed: {url}")
                    print(e)
            data.append(d)
        print(len(data))

df = pd.DataFrame(data)
print(df)
df.to_csv("labeled.csv",index=False)
print("Data saved to labeled.csv")