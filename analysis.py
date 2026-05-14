import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import string
from tqdm import tqdm
from collections import Counter
import spacy
import re
import numpy as np
drive.mount('/content/drive')
nlp = spacy.load("el_core_news_sm")
file_path = '/content/drive/MyDrive/mati.csv'

def load_data(filename):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(filename)

def save_to_csv(df, filename):
    return df.to_csv(filename, index=False)

#Reduces noise in the text data to enhance the quality of subsequent analysis.
def preprocess_text(text):
    # This function preprocesses a Pandas Series of text data by performing the
    # following operations:
    # 1. Converts text to lowercase.
    # 2. Removes mentions (e.g., @username).
    # 3. Removes retweet indicators (e.g., 'RT').
    # 4. Removes links (e.g., http://example.com, www.example.com).
    # 5. Removes only the hashtag mark (#) but keeps the text following it.
    # 6. Replaces words containing punctuation with their non-punctuated versions.
    # 7. Removes ellipsis-like characters (e.g., …).
    # 8. Removes consecutive punctuation marks (e.g., !!!, ??).
    # 9. Removes single punctuation characters such as commas, colons, and hyphens.
    # 10. Removes emoticons and other symbols using Unicode ranges.
    # 11. Removes words with less than 4 characters.
    # 12. Collapses multiple spaces into one and trims leading or trailing spaces.

    # Input Parameters:
    # - text (pd.Series): Pandas Series containing the text to preprocess.

    # Output Parameters:
    # - pd.Series: Cleaned text.

    def remove_punctuation_from_words(line):
        # Splits the line into words and removes punctuation from each word
        words = line.split()
        # Remove punctuation from each word
        cleaned_words = [word.translate(str.maketrans("", "", string.punctuation)) for word in words]
        # Join the cleaned words back into a single string
        return " ".join(cleaned_words)

    # Convert text to lowercase
    text = text.str.lower()
    # Remove mentions (@username)
    text = text.str.replace(r"@\w+", "", regex=True)  # Matches any word starting with '@'
    # Remove retweet indicators (e.g., RT)
    text = text.str.replace(r"\b(rt)\b", "", regex=True)  # Matches 'rt' as a whole word
    # Remove links (e.g., http://example.com or www.example.com)
    text = text.str.replace(r"http\S+|www\S+", "", regex=True)  # Matches URLs starting with http or www
    # Remove only the hashtag mark (#) but keep the text after it
    text = text.str.replace(r"#", "", regex=True)  # Matches the '#' character
    # Replace words containing punctuation with their non-punctuated version
    text = text.apply(remove_punctuation_from_words)  # Applies word-level punctuation removal
    # Remove ellipsis-like characters (…)
    text = text.str.replace(r"…", "", regex=True)  # Matches the Unicode ellipsis character '…'
    # Remove consecutive punctuation marks (e.g., !!!, ??)
    text = text.str.replace(r"[.!?;&]{1,}", "", regex=True)  # Matches one or more consecutive ., !, ;,& or ?
    # Remove single full stops, commas, colons, and hyphens
    text = text.str.replace(r"[.,:\-]", "", regex=True)  # Matches ., :, -, or ,
    # Remove emoticons using Unicode ranges
    text = text.str.replace(
        r"[\U0001F600-\U0001F64F" # Emoticons
        r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        r"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        r"\U00002600-\U000026FF"  # Miscellaneous Symbols
        r"\U00002700-\U000027BF"  # Dingbats
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"]+",
        "",
        regex=True
    )  # Matches Unicode ranges for various emoji and symbols
    # Remove words with less than 4 characters
    text = text.str.replace(r'\b\w{1,3}\b', '', regex=True)  # Matches whole words with length <= 3
    # Remove extra spaces caused by word removal
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()  # Collapses multiple spaces into one and trims
    return text
# =============================================================================

#It keeps only strong tokens for meaning in the text: removes stopwords, non-alphanumeric tokens, and short words (<4 characters).
def clean_tweets(texts,index):
    # This function cleans tweets in bulk using spaCy's nlp.pipe() for efficient
    # processing.

    # Input Parameters:
    # - texts (list of str): List of text entries to clean.

    # Output Parameters:
    # - pd.Series: A pandas Series containing the cleaned text.

    cleaned_texts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=1000, disable=["parser", "ner", "tagger"]),
                    total=len(texts), desc="Cleaning tweets"):
        # Process tokens: retain punctuation, remove stopwords, non-alphanumeric tokens, and short words
        tokens = [
            token.text for token in doc
            if not token.is_stop and token.is_alpha and len(token.text) >= 4
        ]
        cleaned_texts.append(" ".join(tokens))

    # Return the cleaned texts as a pandas Series
    return pd.Series(cleaned_texts, index=index) #prevents pandas index-alignment issues

def use_bag_of_words(df):
    place_pat = r"(" + "|".join([
    r"μάτι", r"ματι",
    r"νέα\s*μάκρη", r"νεα\s*μακρη",
    r"ραφήνα", r"ραφινα",
    r"ανατολικ(ή|η)\s*αττικ(ή|η)",
    r"νέος\s*βουτζάς", r"νεος\s*βουτζας",
    r"κόκκινο\s*λιμανάκι", r"κοκκινο\s*λιμανακι",
    r"ζούμπερι", r"ζουμπερι",
    r"μαραθώνας", r"μαραθωνας",
    r"πικέρμι", r"πικερμι",
    r"πεντέλη", r"πεντελη"
]) + r")"

    event_pat = r"(" + "|".join([
        r"φωτιά", r"φωτια",
        r"πυρκαγι", r"πυρκα", r"πύριν", r"πυριν",
        r"καπν", r"φλογ",
        r"καμεν", r"σταχτ",
        r"καταστροφ", r"τραγωδ", r"φονικ",
        r"νεκρ", r"θυμα", r"θύμα", r"τραυματ",
        r"εκκενωσ", r"πυροσβεσ", r"διασωσ", r"εγκλωβ",
        r"βοήθεια", r"βοηθεια", r"κινδυν",
        r"δικη", r"ευθυν", r"κατηγορ", r"κατηγορουμ",
        r"εγκλημα", r"καταδικ", r"αθω",
        r"επέτει", r"επετει", r"μνημ", r"δε\s*θα\s*ξεχ"
    ]) + r")"

    df_filtered = df[
        df["cleaned_text"].str.contains(place_pat, regex=True, na=False) &
        df["cleaned_text"].str.contains(event_pat, regex=True, na=False)
]


    return df_filtered

def fill_missing_dates(daily_volume_df):
    # This function identifies missing dates in the dataset, adds them with
    # zero volume, and reports the number of new dates added.

    # Input Parameters:
    # - daily_volume_df (DataFrame): A DataFrame containing:
    #                                - 'date': A column of timestamps
    #                                          representing dates.
    #                                - 'volume': A column of volumes
    #                                            corresponding to each date.

    # Output Parameters:
    # - updated_df (DataFrame): The updated DataFrame with all missing dates
    #                           filled with zero volume.

    # Determine the range of dates
    earliest_date = daily_volume_df['dates'].min()
    latest_date = daily_volume_df['dates'].max()

    # Generate a complete date range from earliest to latest date
    complete_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')

    # Create a DataFrame for the complete date range
    all_dates_df = pd.DataFrame({'dates': complete_date_range})

    # Merge with the existing DataFrame to identify missing dates
    # Ensure both columns are of the same type (datetime64[ns])
    daily_volume_df['dates'] = daily_volume_df['dates'].astype('datetime64[ns]')
    all_dates_df['dates'] = all_dates_df['dates'].astype('datetime64[ns]')

    updated_df = pd.merge(all_dates_df, daily_volume_df, on='dates', how='left')

    # Fill missing volumes with zero
    updated_df['volume'] = updated_df['volume'].fillna(0).astype(int)

    # Report the number of missing dates added
    new_dates_count = len(complete_date_range) - len(daily_volume_df)
    print(f"Number of missing dates added: {new_dates_count}")

    # Ensure the 'dates' column contains only the date component
    #updated_df['dates'] = updated_df['dates'].dt.date

    return updated_df

def plot_daily_volume(daily_volume_df, day_min, day_max):
    # This function generates a plot of daily tweet volumes for a specified
    # period, with validation.

    # Input Parameters:
    # - daily_volume_df (DataFrame): A DataFrame containing a 'date' column
    #                                (timestamps of dates)  and a 'volume'
    #                                column (corresponding daily volume).
    # - day_min (int): Starting day (1-indexed).
    # - day_max (int): Ending day (1-indexed).

    # The function ensures the provided day range is valid and plots the daily
    # tweet volumes for the specified range.

    # Validate day_min and day_max
    total_days = len(daily_volume_df)
    if day_min < 1 or day_max > total_days or day_min > day_max:
        raise ValueError(
            f"Invalid day range: day_min={day_min}, day_max={day_max}. "
            f"Valid range is 1 to {total_days}, and day_min should not exceed day_max."
        )

    # Extract the subset of data based on the day range
    daily_volume_subset = daily_volume_df.iloc[day_min - 1:day_max]

    figure, axes = plt.subplots(1,1)
    plt.style.use('dark_background')
    axes.plot(daily_volume_subset['volume'], color='red', alpha=0.8)
    axes.set_xlabel('Time since accident', color='white')
    axes.set_ylabel('Daily tweet volume', color='white')
    axes.set_title(f'Daily tweet volume {day_min} to {day_max} day', color='white')
    # Add grid
    axes.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)

    # Customize tick colors for dark background
    axes.tick_params(axis='x', colors='white')
    axes.tick_params(axis='y', colors='white')

    # Show the plot
    plt.show()

def add_rolling_stats(daily_volume_df, window_size=30):
    daily_volume_df = daily_volume_df.copy()
    daily_volume_df["dates"] = pd.to_datetime(daily_volume_df["dates"])

    shifted = daily_volume_df["volume"].shift(1)
    daily_volume_df["rolling_mean"] = (
        shifted.rolling(window=window_size, min_periods=1).mean().fillna(0)
    )
    daily_volume_df["rolling_std"] = (
        shifted.rolling(window=window_size, min_periods=1).std().fillna(0)
    )
    return daily_volume_df

def detect_bursts(daily_volume_df, K=2):
    daily_volume_df["burst_threshold"] = (
        daily_volume_df["rolling_mean"] +
        K * daily_volume_df["rolling_std"]
    )

    daily_volume_df["is_burst"] = (
        daily_volume_df["volume"] >
        daily_volume_df["burst_threshold"]
    )

    return daily_volume_df


def plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2):
    """
    Plots daily volume (index on x-axis) + rolling mean + burst threshold
    and highlights detected burst days.

    Requirements:
    daily_volume_df must contain: 'volume', 'rolling_mean', 'rolling_std'
    (or you compute them before calling), and it will use 'is_burst' and
    'burst_threshold' if present; otherwise it computes them.
    """
    # Validate day_min and day_max (1-indexed)
    total_days = len(daily_volume_df)
    if day_min < 1 or day_max > total_days or day_min > day_max:
        raise ValueError(
            f"Invalid day range: day_min={day_min}, day_max={day_max}. "
            f"Valid range is 1 to {total_days}, and day_min should not exceed day_max."
        )

    # Subset
    subset = daily_volume_df.iloc[day_min - 1:day_max].copy()

    # If rolling stats not present, compute them (causal rolling)
    if "rolling_mean" not in subset.columns or "rolling_std" not in subset.columns:
        roll = subset["volume"].rolling(window=window_size, min_periods=1)
        subset["rolling_mean"] = roll.mean()
        subset["rolling_std"] = roll.std().fillna(0.0)

    # Compute threshold + burst flags if not present
    if "burst_threshold" not in subset.columns:
        subset["burst_threshold"] = subset["rolling_mean"] + K * subset["rolling_std"]

    if "is_burst" not in subset.columns:
        subset["is_burst"] = subset["volume"] > subset["burst_threshold"]

    # X axis = index (days)
    x = subset.index

    figure, axes = plt.subplots(1, 1, figsize=(12, 6))
    plt.style.use("dark_background")

    # Daily volume
    axes.plot(x, subset["volume"], color="red", alpha=0.8, label="Daily Volume")

    # Rolling mean
    axes.plot(x, subset["rolling_mean"], color="yellow", linewidth=2,
              label=f"{window_size}-Day Rolling Mean")

    # Threshold
    axes.plot(x, subset["burst_threshold"], color="white", linestyle="--", linewidth=1.5,
              label=f"Threshold (mean + {K}σ)")

    # Burst markers
    bursts = subset[subset["is_burst"]]
    axes.scatter(bursts.index, bursts["volume"], color="cyan", s=25, alpha=0.9,
                 label="Detected Bursts", zorder=3)

    axes.set_xlabel("Time since accident (days)", color="white")
    axes.set_ylabel("Daily tweet volume", color="white")
    axes.set_title("Daily tweet volume with burst detection", color="white")

    # Grid + ticks (same as your style)
    axes.grid(color="yellow", linestyle="--", linewidth=0.5, alpha=0.7)
    axes.tick_params(axis="x", colors="white")
    axes.tick_params(axis="y", colors="white")

    axes.legend()
    plt.tight_layout()
    plt.show()

def plot_author_volume(tweets_df,day_min,day_max):
    # This function computes the volume of tweets per author for a given range
    # of days and generates the corresponding histogram plot.

    # Input Parameters:
    # - tweets_df (DataFrame): A DataFrame containing:
    #                          - 'author_id': ID of the author.
    #                          - 'dates': Dates of the tweets (datetime format).
    # - day_min (int): The starting day (1-indexed) from the earliest date.
    # - day_max (int): The ending day (inclusive, 1-indexed).

    # Output Parameters:
    # - result_df (DataFrame): A DataFrame with columns:
    #                          - 'author_id': The unique author IDs.
    #                          - 'volume': The number of tweets per author in
    #                             the given range.

    # Get the earliest and latest dates
    earliest_date = tweets_df["dates"].min()
    latest_date = tweets_df["dates"].max()

    # Total number of days in the dataset
    total_days = (latest_date - earliest_date).days + 1

    # Validate the specified range
    if day_min < 1 or day_max > total_days or day_min > day_max:
        raise ValueError(
            f"Invalid day range: day_min={day_min}, day_max={day_max}. "
            f"Valid range is from 1 to {total_days} days."
        )

    # Calculate the date range corresponding to day_min and day_max
    start_date = earliest_date + pd.Timedelta(days=day_min - 1)
    end_date = earliest_date + pd.Timedelta(days=day_max - 1)

    # Filter the dataset for the specified range of days
    # Αν η στήλη είναι UTC
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    filtered_data = tweets_df[(tweets_df["timestamps"] >= start_date) &
                          (tweets_df["timestamps"] <= end_date)]




    # Group by author_id and count the tweets
    author_volume_df = filtered_data.groupby("author_id").size().reset_index(name="volume")

    # Sort by tweet volume in descending order
    author_volume_df = author_volume_df.sort_values(by="volume", ascending=False).reset_index(drop=True)

    # Isolate the volume of tweets per author
    author_volume = author_volume_df["volume"]

    # Generate the corresponding author volume range list.
    author_volume_bins = [v for v in range(min(author_volume),max(author_volume)+1)]

    # Set the largest k volumes of tweets to be represented in the frequency histogram
    low_k_volumes = 60
    # Set up a figure with a set of axes
    figure, axes = plt.subplots(1, 1)
    # Set a dark background
    plt.style.use('dark_background')
    # Generate the histogram
    axes.hist(author_volume, bins=author_volume_bins[:low_k_volumes], color="red", alpha=0.7)
    # Add axes labels
    axes.set_xlabel('Tweets Volume', color='white')
    axes.set_ylabel('Number of Authors', color='white')
    # Set the title string.
    title_string = f"Distribution of Authors per Number of Tweets from Day {day_min} to Day {day_max}"
    # Add title to the figure
    axes.set_title(title_string, color='white')
    # Add grid
    axes.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    # Customize tick colors for dark background
    axes.tick_params(axis='x', colors='white')
    axes.tick_params(axis='y', colors='white')
    # Show the plot
    plt.show()

    return author_volume_df
def calculate_author_timespans(filtered_df):
    author_timespans = filtered_df.groupby('author_id')['timestamps'].agg(
        lambda x: (x.max() - x.min()).days
    )
    return author_timespans

def calculate_author_statistics(filtered_df):
    df = filtered_df[['author_id', 'timestamps']].copy()

    #Be sure timestamps are in datetime format and sort by author and timestamp
    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    df = df.dropna(subset=["timestamps"]).sort_values(["author_id", "timestamps"])

    #Find first tweet, last tweet, total tweets per author
    author_stats = (
        df.groupby('author_id')['timestamps']
        .agg(first_tweet='min', last_tweet='max', total_tweets='count')
        .reset_index()
    )
    #Time span between first and last tweet in hours
    span = author_stats['last_tweet'] - author_stats['first_tweet']
    author_stats['timespan_hours'] = span.dt.total_seconds() / 3600

    #Since timespans are sorted, we can calculate average interval between tweets
    df['delta_hours'] = df.groupby('author_id')['timestamps'].diff().dt.total_seconds() / 3600

    avg_interval = df.groupby('author_id')['delta_hours'].mean()
    #For every author, map the average interval back to author_stats
    author_stats['avg_interval_hours'] = author_stats['author_id'].map(avg_interval)

    return author_stats

def robust_minmax(s):
    lo = s.quantile(0.05)
    hi = s.quantile(0.99)
    #here we check if hi and lo are very close to avoid division by zero. If they are., it means that all values are almost the same, so they dont carry any information.
    if np.isclose(hi, lo):
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def feature_engineering_authors(author_stats):
    df = author_stats.copy()
    df['frequency'] = 1 / (df['avg_interval_hours'] + 1)  # Avoid division by zero
    df['impact'] = df['total_tweets'] / (df['timespan_hours'] + 1)  # Avoid division by zero
    df['frequency'] = df['frequency'].fillna(0.0) #avoid breaking for users with single tweet
    df['impact'] = df['impact'].fillna(0.0) #avoid breaking for users with single tweet
    df['norm_total_tweets'] = robust_minmax(df['total_tweets'])
    df['norm_frequency'] = robust_minmax(df['frequency'])
    df['norm_impact'] = robust_minmax(df['impact'])

    return df

def rank_authors(author_stats, w_total=0.4, w_freq=0.3, w_impact=0.3):
    df = author_stats.copy()
    df['activity_score'] = (
        w_total * df['norm_total_tweets'] +
        w_freq * df['norm_frequency'] +
        w_impact * df['norm_impact']
    )
    df = df.sort_values(by='activity_score', ascending=False).reset_index(drop=True)
    return df

def plot_top_authors(ranked_authors, top_n=20):
    top = ranked_authors.head(top_n).copy()
    x = np.arange(len(top))

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(x, top["activity_score"], alpha=0.8)
    plt.xticks(x, top["author_id"].astype(str), rotation=60, ha="right", color="white")
    plt.xlabel("Author ID", color="white")
    plt.ylabel("Activity Score", color="white")
    plt.title(f"Top {top_n} Most Active Authors", color="white")
    plt.grid(color="yellow", linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
    plt.tick_params(axis="y", colors="white")
    plt.tight_layout()
    plt.show()

def plot_hour_of_day_activity(filtered_df):
    df = filtered_df.copy()

    df['hour'] = df['timestamps'].dt.hour
    df['date'] = df['timestamps'].dt.date

    hour_volume = df.groupby(['date', 'hour']).size().reset_index(name='volume')
    hour_avg = hour_volume.groupby('hour')['volume'].mean().reset_index(name='avg_volume')
    avg_line = hour_avg['avg_volume'].mean()

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(hour_avg['hour'], hour_avg['avg_volume'], color='red', alpha=0.7)
    plt.axhline(y=avg_line, color='yellow', linestyle='--', label='Average Volume')
    plt.xlabel('Hour of Day', color='white')
    plt.ylabel('Avg Tweets per Day', color='white')
    plt.title('Tweet Volume by Hour of Day', color='white')
    plt.xticks(range(0, 24), color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_engagement_metrics(filtered_df):
    df = filtered_df.copy()

    engagement = (
        df.groupby('author_id')
        .agg(
            total_likes=('like_count', 'sum'),
            total_replies=('reply_count', 'sum'),
            total_retweets=('retweet_count', 'sum'),
            total_tweets=('tweet_id', 'count')
        )
        .reset_index()
    )

    engagement['avg_engagement_tweet'] = (
        engagement['total_likes'] +
        engagement['total_replies'] +
        engagement['total_retweets'] ) / engagement['total_tweets']

    engagement['total_engagement'] = (
        engagement['total_likes'] +
        engagement['total_replies'] +
        engagement['total_retweets']
    )
    return engagement

def rank_authors_by_engagement(engagement_df, top_n=20, sorting_metric='avg_engagement_tweet'):
    df = engagement_df.copy()
    df = df.sort_values(by=sorting_metric, ascending=False).reset_index(drop=True)
    top_authors = df.head(top_n)

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(top_authors['author_id'].astype(str), top_authors[sorting_metric], color='red', alpha=0.8)
    plt.xlabel('Author ID', color='white')
    plt.ylabel(f'{sorting_metric}', color='white')
    plt.title(f'Top {top_n} Authors by {sorting_metric}', color='white')
    plt.xticks(rotation=60, ha='right', color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()
	##ANASTASIA
def analyze_time_gaps(filtered_df):
    df = filtered_df[['author_id', 'timestamps']].copy()
    df = df.sort_values(['author_id', 'timestamps'])

    df['gap_seconds'] = (
        df.groupby('author_id')['timestamps']
        .diff()
        .dt.total_seconds()
    )

    stats = (
        df.groupby('author_id')['gap_seconds']
        .agg(
            avg_gap_seconds='mean',
            std_gap_seconds='std',
            short_gaps=lambda x: (x < 60).sum(),
            long_gaps=lambda x: (x > 86400).sum()
        )
        .reset_index()
    )

    return stats
def identify_irregular_authors(gap_stats,
                               short_gap_threshold=10,
                               long_gap_threshold=10):
    """
    Identifies authors with irregular tweeting patterns.
    """
    irregular = gap_stats[
        (gap_stats['short_gaps'] >= short_gap_threshold) |
        (gap_stats['long_gaps'] >= long_gap_threshold)
    ].sort_values(
        by=['short_gaps', 'long_gaps'],
        ascending=False
    )
    return irregular##Q6


def compute_weekly_totals(filtered_df):
    df = filtered_df.copy()
    iso = df['timestamps'].dt.isocalendar()

    df['iso_year'] = iso.year
    df['iso_week'] = iso.week

    weekly_totals = (
        df.groupby(['iso_year', 'iso_week'])
        .size()
        .reset_index(name='total_volume')
    )

    weekly_totals['year_week'] = (
        weekly_totals['iso_year'].astype(str) + '-W' + weekly_totals['iso_week'].astype(str)
    )

    return weekly_totals
def compute_weekly_totals(filtered_df):
    df = filtered_df.copy()
    df['week'] = df['timestamps'].dt.isocalendar().week
    weekly_totals = df.groupby('week').size().reset_index(name='total_volume')
    return weekly_totals

# Q8. Author Retweet Dependency
# =========================

def compute_retweet_dependency(filtered_df):
    df = filtered_df.copy()

    # Use raw_text, not cleaned/preprocessed text
    df['is_retweet'] = df['raw_text'].astype(str).str.match(r'(?i)^rt\s+')

    stats = (
        df.groupby('author_id')
        .agg(
            total_tweets=('tweet_id', 'count'),
            retweets=('is_retweet', 'sum')
        )
        .reset_index()
    )

    stats['retweets'] = stats['retweets'].astype(int)
    stats['original_tweets'] = stats['total_tweets'] - stats['retweets']
    stats['retweet_ratio'] = stats['retweets'] / stats['total_tweets']

    stats['retweet_to_original_ratio'] = np.where(
        stats['original_tweets'] > 0,
        stats['retweets'] / stats['original_tweets'],
        np.nan
    )

    return stats


def plot_retweet_dependency_distribution(retweet_stats):
    df = retweet_stats.copy()

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.hist(df['retweet_ratio'].dropna(), bins=30, alpha=0.8)
    plt.xlabel('Retweet Ratio', color='white')
    plt.ylabel('Number of Authors', color='white')
    plt.title('Distribution of Author Retweet Dependency', color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.show()


def rank_authors_by_retweet_dependency(retweet_stats, top_n=20, metric='retweet_ratio'):
    df = retweet_stats.copy()
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    top = df.head(top_n)

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(top['author_id'].astype(str), top[metric], alpha=0.8)
    plt.xlabel('Author ID', color='white')
    plt.ylabel(metric, color='white')
    plt.title(f'Top {top_n} Authors by {metric}', color='white')
    plt.xticks(rotation=60, ha='right', color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()

    return top
##Q9
# =========================
# Q9. Burst-Origin Analysis
# =========================

def burst_origin_analysis(filtered_df, daily_volume_df, hours_before=6):
    """
    For each burst day:
    1. Examine the preceding time window (e.g. 6 hours before burst onset)
    2. Examine tweets during the burst day itself
    3. Compute:
       - pct_preburst_window
       - pct_burst_day
       - lag_hours
    """

    bursts_df = daily_volume_df[daily_volume_df['is_burst']].copy()
    results = []

    for _, row in bursts_df.iterrows():
        burst_date = pd.Timestamp(row['dates'])
        burst_start = burst_date
        burst_end = burst_start + pd.Timedelta(days=1)
        window_start = burst_start - pd.Timedelta(hours=hours_before)

        preburst_tweets = filtered_df[
            (filtered_df['timestamps'] >= window_start) &
            (filtered_df['timestamps'] < burst_start)
        ].copy()

        burst_day_tweets = filtered_df[
            (filtered_df['timestamps'] >= burst_start) &
            (filtered_df['timestamps'] < burst_end)
        ].copy()

        total_preburst = len(preburst_tweets)
        total_burst_day = len(burst_day_tweets)

        authors = sorted(
            set(preburst_tweets['author_id'].unique()).union(
                set(burst_day_tweets['author_id'].unique())
            )
        )

        for author in authors:
            author_pre = preburst_tweets[preburst_tweets['author_id'] == author]
            author_burst = burst_day_tweets[burst_day_tweets['author_id'] == author]

            pre_count = len(author_pre)
            burst_count = len(author_burst)

            first_pre_tweet = author_pre['timestamps'].min() if pre_count > 0 else pd.NaT
            lag_hours = (
                (burst_start - first_pre_tweet).total_seconds() / 3600
                if pd.notna(first_pre_tweet) else np.nan
            )

            results.append({
                'burst_date': burst_date,
                'author_id': author,
                'tweets_preburst_window': pre_count,
                'tweets_burst_day': burst_count,
                'pct_preburst_window': pre_count / total_preburst if total_preburst > 0 else 0.0,
                'pct_burst_day': burst_count / total_burst_day if total_burst_day > 0 else 0.0,
                'lag_hours': lag_hours
            })

    return pd.DataFrame(results)


def rank_burst_drivers(burst_authors, top_n=20, metric='pct_burst_day'):
    df = burst_authors.copy()
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    top = df.head(top_n)

    print(f"Top {top_n} burst drivers by {metric}:")
    print(top[['burst_date', 'author_id', 'tweets_preburst_window', 'tweets_burst_day', metric, 'lag_hours']])

    return top


def plot_burst_driver_contributions(burst_authors, top_n=15, metric='pct_burst_day'):
    df = (
        burst_authors.groupby('author_id')[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(df['author_id'].astype(str), df[metric], alpha=0.8)
    plt.xlabel('Author ID', color='white')
    plt.ylabel(f'Average {metric}', color='white')
    plt.title(f'Top {top_n} Authors by Average Burst Contribution', color='white')
    plt.xticks(rotation=60, ha='right', color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()
    # =========================
# Q10. Identification of Potentially Influential / Manipulative Authors
# =========================

def identify_influential_authors(
    ranked_authors,
    engagement_df,
    retweet_stats,
    burst_authors,
    gap_stats
):
    burst_summary = (
        burst_authors.groupby('author_id')
        .agg(
            avg_preburst_contribution=('pct_preburst_window', 'mean'),
            avg_burst_contribution=('pct_burst_day', 'mean'),
            bursts_participated=('burst_date', 'nunique'),
            avg_lag_hours=('lag_hours', 'mean'),
            median_lag_hours=('lag_hours', 'median')
        )
        .reset_index()
    )

    df = ranked_authors.merge(engagement_df, on='author_id', how='left')
    df = df.merge(retweet_stats, on='author_id', how='left')
    df = df.merge(gap_stats, on='author_id', how='left')
    df = df.merge(burst_summary, on='author_id', how='left')

    df['high_retweet_dependency_flag'] = df['retweet_ratio'] > 0.8
    df['high_short_gap_flag'] = df['short_gaps'] >= 5
    df['early_burst_flag'] = df['median_lag_hours'] <= 2
    df['repeat_burst_presence_flag'] = df['bursts_participated'] >= 2

    df['influence_score'] = (
        0.25 * df['activity_score'].fillna(0) +
        0.20 * robust_minmax(df['avg_engagement_tweet'].fillna(0)) +
        0.25 * robust_minmax(df['avg_burst_contribution'].fillna(0)) +
        0.15 * robust_minmax(df['bursts_participated'].fillna(0)) +
        0.15 * (
            1 - robust_minmax(
                df['median_lag_hours'].fillna(
                    df['median_lag_hours'].max() if df['median_lag_hours'].notna().any() else 0
                )
            )
        )
    )

    df['coordination_risk_score'] = (
        0.35 * robust_minmax(df['retweet_ratio'].fillna(0)) +
        0.25 * robust_minmax(df['short_gaps'].fillna(0)) +
        0.20 * robust_minmax(df['bursts_participated'].fillna(0)) +
        0.20 * (
            1 - robust_minmax(
                df['median_lag_hours'].fillna(
                    df['median_lag_hours'].max() if df['median_lag_hours'].notna().any() else 0
                )
            )
        )
    )

    df = df.sort_values(
        ['influence_score', 'coordination_risk_score'],
        ascending=False
    ).reset_index(drop=True)

    return df


def plot_influential_authors(influential_df, top_n=15, metric='influence_score'):
    top = influential_df.head(top_n)

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(top['author_id'].astype(str), top[metric], alpha=0.8)
    plt.xlabel('Author ID', color='white')
    plt.ylabel(metric, color='white')
    plt.title(f'Top {top_n} Authors by {metric}', color='white')
    plt.xticks(rotation=60, ha='right', color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()


def plot_coordination_risk(influential_df, top_n=15):
    top = influential_df.sort_values('coordination_risk_score', ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")
    plt.bar(top['author_id'].astype(str), top['coordination_risk_score'], alpha=0.8)
    plt.xlabel('Author ID', color='white')
    plt.ylabel('coordination_risk_score', color='white')
    plt.title(f'Top {top_n} Authors by Coordination Risk Score', color='white')
    plt.xticks(rotation=60, ha='right', color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7, axis='y')
    plt.tight_layout()
    plt.show()


def plot_author_burst_timeline(filtered_df, burst_authors, author_id, hours_before=6, hours_after=24):
    df = filtered_df[filtered_df['author_id'] == author_id].copy()
    burst_dates = burst_authors[burst_authors['author_id'] == author_id]['burst_date'].dropna().unique()

    if len(burst_dates) == 0:
        print(f"No burst-related activity found for author {author_id}")
        return

    plt.figure(figsize=(12, 6))
    plt.style.use("dark_background")

    for burst_date in burst_dates[:5]:
        burst_start = pd.Timestamp(burst_date)
        start = burst_start - pd.Timedelta(hours=hours_before)
        end = burst_start + pd.Timedelta(hours=hours_after)

        author_window = df[(df['timestamps'] >= start) & (df['timestamps'] < end)].copy()
        if author_window.empty:
            continue

        relative_hours = (author_window['timestamps'] - burst_start).dt.total_seconds() / 3600
        plt.scatter(
            relative_hours,
            np.ones(len(relative_hours)) * burst_start.day,
            alpha=0.7,
            label=str(burst_date.date())
        )

    plt.axvline(0, linestyle='--')
    plt.xlabel('Hours relative to burst onset', color='white')
    plt.ylabel('Burst day marker', color='white')
    plt.title(f'Activity Timeline for Author {author_id} Around Bursts', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_weekly_heatmap(weekly_df):
      pivot = weekly_df.pivot(index='week', columns='weekday', values='volume').fillna(0)
      plt.figure(figsize=(12, 8))
      plt.style.use("dark_background")
      plt.imshow(pivot, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
      plt.colorbar(label='Tweet Volume')
      plt.xlabel('Day of Week')
      plt.ylabel('Week of Year')
      plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
      plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
      plt.title('Weekly Tweet Activity Heatmap')
      plt.tight_layout()
      plt.show()
def compute_weekly_activity(filtered_df):
    df = filtered_df.copy()

    df['week'] = df['timestamps'].dt.isocalendar().week
    df['weekday'] = df['timestamps'].dt.dayofweek  # Monday=0

    weekly_df = (
        df.groupby(['week', 'weekday'])
        .size()
        .reset_index(name='volume')
    )

    return weekly_df

def main():
    filename = '/content/drive/MyDrive/mati.csv'
    tweets_df = load_data(filename)
    #Define columns
    tweets_df.columns = ['author_id', 'created_at', 'geo', 'tweet_id', 'lang', 'like_count', 'quote_count', 'reply_count', 'retweet_count', 'source', 'text']
    #Convert created_at to date format
    tweets_df['timestamps'] = pd.to_datetime(tweets_df['created_at'], utc=True).dt.tz_localize(None)
    tweets_df['dates'] = tweets_df['timestamps'].dt.date
    #Q1: Preprocess and clean tweets, then filter using bag-of-words model
    tweets_df = tweets_df.sort_values(by='timestamps', ascending=True)

    # κρατάμε το αρχικό text για Q8
    tweets_df["raw_text"] = tweets_df["text"].astype(str)
    # preprocess μόνο για filtering / text analysis
    tweets_df["text"] = preprocess_text(tweets_df["raw_text"])
    tweets_df["cleaned_text"] = clean_tweets(tweets_df["text"].tolist(), tweets_df.index)
    filtered_df = use_bag_of_words(tweets_df)
    remaining_df = tweets_df.drop(filtered_df.index) # Get tweets that were not selected for further analysis

    print(f"Original dataset tweets count: {len(tweets_df)}")
    print(f"Filtered dataset tweets count: {len(filtered_df)}")
    print(f"Irrelevant dataset tweets count: {len(remaining_df)}")

    # Save filtered and remaining tweets to separate CSV files
    save_to_csv(filtered_df, 'filtered_tweets.csv')
    save_to_csv(remaining_df, 'remaining_tweets.csv')

    #Q2: Analyze daily tweet volume, plot results, identify bursts
    daily_volume_df = filtered_df.groupby('dates').size().reset_index(name='volume')
    daily_volume = daily_volume_df['volume']
    # Calculate the time period in days that spans the dataset.
    earliest_date = filtered_df["dates"].min()
    latest_date = filtered_df["dates"].max()
    days_period = (latest_date - earliest_date).days
    # Calculate the number of dates in the dataset with non-zero volume of tweets.
    non_zero_days = len(daily_volume)
    total_days = (latest_date - earliest_date).days + 1

    # Report the earliest and latest dates in the dataset alonside with the total
    # time duration in days.
    print(f"Earliest date: {earliest_date}")
    print(f"Latest date: {latest_date}")
    print(f"Time period in days: {days_period}")
    print(f"Number of non-zero volume days: {non_zero_days}")

    # Update the daily volume dataframe to incorporate the missing dates with zero
    # volume entries.
    daily_volume_df = fill_missing_dates(daily_volume_df)
    daily_volume = daily_volume_df['volume']
    # Plot the daily volume of tweets since day zero for the complete time range of
    # the dataset measured in days.
    # plot daily volume in segments of 1 year (365 days)
    day_min = 1
    day_max = 365
    plot_daily_volume(daily_volume_df, day_min, day_max)
    daily_volume_df = add_rolling_stats(daily_volume_df, 30)
    daily_volume_df = detect_bursts(daily_volume_df, K=2)
    plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2)

    day_min = 365
    day_max = 2 * 365
    plot_daily_volume(daily_volume_df, day_min, day_max)
    daily_volume_df = add_rolling_stats(daily_volume_df, 30)
    daily_volume_df = detect_bursts(daily_volume_df, K=2)
    plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2)

    day_min = 2 * 365
    day_max = 3 * 365
    plot_daily_volume(daily_volume_df, day_min, day_max)
    daily_volume_df = add_rolling_stats(daily_volume_df, 30)
    daily_volume_df = detect_bursts(daily_volume_df, K=2)
    plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2)

    day_min = 3 * 365
    day_max = 4 * 365
    plot_daily_volume(daily_volume_df, day_min, day_max)
    daily_volume_df = add_rolling_stats(daily_volume_df, 30)
    daily_volume_df = detect_bursts(daily_volume_df, K=2)
    plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2)

    day_min = 4 * 365
    day_max = len(daily_volume_df)
    plot_daily_volume(daily_volume_df, day_min, day_max)
    daily_volume_df = add_rolling_stats(daily_volume_df, 30)
    daily_volume_df = detect_bursts(daily_volume_df, K=2)
    plot_bursts_indexed(daily_volume_df, day_min, day_max, window_size=30, K=2)


    # Q3: Analyze author activity and rank authors
    day_min = 1
    day_max = total_days
    author_volume_df = plot_author_volume(filtered_df, day_min, day_max)
    author_stats = calculate_author_statistics(filtered_df)
    engineered_stats = feature_engineering_authors(author_stats)
    ranked_authors = rank_authors(engineered_stats)
    print(ranked_authors.head(10))
    plot_top_authors(ranked_authors, top_n=20)

    #Q4: Time-of-day activity analysis
    plot_hour_of_day_activity(filtered_df)

    #Q5: Engagement metrics and ranking
    engagement_df = compute_engagement_metrics(filtered_df)
    print(engagement_df.head(10))
    rank_authors_by_engagement(engagement_df, top_n=20, sorting_metric='avg_engagement_tweet')
    rank_authors_by_engagement(engagement_df, top_n=20, sorting_metric='total_engagement')
 #  Q6: Time-gap analysis
    gap_stats = analyze_time_gaps(filtered_df)
    irregular_authors = identify_irregular_authors(gap_stats)

    print("Irregular authors (top 10):")
    print(irregular_authors.head(10))
#Q7
    weekly_df = compute_weekly_activity(filtered_df)
    weekly_totals = compute_weekly_totals(filtered_df)

    print("Weekly total activity:")
    print(weekly_totals.head())

    plot_weekly_heatmap(weekly_df)


# Q8
    retweet_stats = compute_retweet_dependency(filtered_df)
    print("Top authors by retweet ratio:")
    print(retweet_stats.sort_values('retweet_ratio', ascending=False).head(10))
    print("Top authors by retweet-to-original ratio:")
    print(retweet_stats.sort_values('retweet_to_original_ratio', ascending=False).head(10))
    plot_retweet_dependency_distribution(retweet_stats)
    rank_authors_by_retweet_dependency(retweet_stats, top_n=20, metric='retweet_ratio')

  # Q9
    burst_authors = burst_origin_analysis(filtered_df, daily_volume_df, hours_before=6)
    print("Top burst drivers:")
    print(
      burst_authors.sort_values('pct_burst_day', ascending=False)
      [['burst_date', 'author_id', 'pct_preburst_window', 'pct_burst_day', 'lag_hours']]
      .head(10)
  )
    rank_burst_drivers(burst_authors, top_n=20, metric='pct_burst_day')
    plot_burst_driver_contributions(burst_authors, top_n=15, metric='pct_burst_day')

  # Q10
    influential = identify_influential_authors(
      ranked_authors,
      engagement_df,
      retweet_stats,
      burst_authors,
      gap_stats
  )

    print("Top influential authors:")
    print(
      influential[
          ['author_id', 'influence_score', 'coordination_risk_score',
          'avg_burst_contribution', 'bursts_participated',
          'median_lag_hours', 'retweet_ratio', 'short_gaps']
      ].head(10)
  )

    plot_influential_authors(influential, top_n=15, metric='influence_score')
    plot_coordination_risk(influential, top_n=15)

    if not influential.empty:
      top_author = influential.iloc[0]['author_id']
      plot_author_burst_timeline(filtered_df, burst_authors, top_author)



if __name__ == "__main__":
    main()