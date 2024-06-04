import praw
import pandas as pd
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import unicodedata

class RedditCommentFetcher:
    def __init__(self, client_id, client_secret, user_agent, submission_ids, output_csv):
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)
        self.submission_ids = submission_ids
        self.output_csv = output_csv

    def normalize_text(self, text):
        return unicodedata.normalize('NFKC', text)

    def fetch_top_level_comments(self, premise_id, submission):
        submission.comments.replace_more(limit=None)
        comments_metadata = []
        for comment in submission.comments.list():
            if comment.is_root and comment.author and comment.author.name not in ['AutoModerator', 'Deleted']:
                comments_metadata.append({
                    'subreddit': submission.subreddit.display_name,
                    'premise_id': premise_id,
                    'post_id': submission.id,
                    'post_title': self.normalize_text(submission.title),
                    'post_description': self.normalize_text(submission.selftext),
                    'post_score': submission.score,
                    'post_created_utc': datetime.fromtimestamp(submission.created_utc),
                    'comment_id': comment.id,
                    'comment_body': self.normalize_text(comment.body),
                    'comment_score': comment.score,
                    'user_id': comment.author.name if comment.author else 'Deleted',
                    'comment_created_utc': datetime.fromtimestamp(comment.created_utc),
                })
        return comments_metadata

    def process_posts(self):
        comments_data = []

        for premise_id, post_id in tqdm(enumerate(self.submission_ids), total=len(self.submission_ids)):
            try:
                submission = self.reddit.submission(id=post_id)
                comments_metadata = self.fetch_top_level_comments(premise_id, submission)
                
                if comments_metadata:
                    # Find the top comment by upvotes
                    top_comment = max(comments_metadata, key=lambda x: x['comment_score'])
                    top_comment_time = top_comment['comment_created_utc']
                    
                    # Define the time window
                    start_time = top_comment_time - timedelta(hours=48)
                    end_time = top_comment_time + timedelta(hours=6)
                    
                    # Filter comments within the window
                    filtered_comments = [comment for comment in comments_metadata if start_time <= comment['comment_created_utc'] <= end_time]
                    comments_data.extend(filtered_comments)
                else:
                    print(f"No valid comments found for post ID '{post_id}'")
            except Exception as e:
                print(f"Error fetching comments for post ID '{post_id}': {e}")

        comments_df = pd.DataFrame(comments_data)
        comments_df.to_csv(self.output_csv, index=False, encoding='utf-8')
        print(f"Comments and metadata have been saved to {self.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch top-level comments from Reddit posts based on submission IDs.")
    parser.add_argument('--client_id', type=str, required=True, help='Reddit API client ID')
    parser.add_argument('--client_secret', type=str, required=True, help='Reddit API client secret')
    parser.add_argument('--user_agent', type=str, required=True, help='Reddit API user agent')
    parser.add_argument('--submission_ids', type=str, required=True, help='Comma-separated list of Reddit submission IDs')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file to save comments')

    args = parser.parse_args()

    submission_ids = args.submission_ids.split(',')

    fetcher = RedditCommentFetcher(client_id=args.client_id,
                                   client_secret=args.client_secret,
                                   user_agent=args.user_agent,
                                   submission_ids=submission_ids,
                                   output_csv=args.output_csv)
    fetcher.process_posts()
