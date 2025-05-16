from Dail_Dataset import Dail_dataset
import matplotlib.pyplot as plt
import pandas as pd 

def graph_word_count_over_time(dataset):
    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()

    # Convert the 'date' column to datetime format and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample by week and sum the word counts
    weekly_word_count = df['word_count'].resample('ME').sum().reset_index()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_word_count['date'], weekly_word_count['word_count'], marker='o')
    plt.title('Word Count Over Time (Monthly Aggregation)')
    plt.xlabel('Date')
    plt.ylabel('Total Word Count')
    plt.grid()
    plt.savefig("word_count_over_time.png")

def top_average_word_count_per_speaker(dataset):
    df = dataset.to_pandas()
    # Group by speaker and sum the word counts
    speaker_word_count = df.groupby('speaker')['word_count'].sum().reset_index()
    # number of speeches for that speaker
    speaker_word_count['num_speeches'] = df.groupby('speaker')['word_count'].count().values
    # Calculate the average word count per speech for each speaker
    speaker_word_count['avg_word_count'] = speaker_word_count['word_count'] / speaker_word_count['num_speeches']
    # Sort speakers by average word count in descending order
    speaker_word_count = speaker_word_count.sort_values('avg_word_count', ascending=False).head(20)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(speaker_word_count['speaker'], speaker_word_count['avg_word_count'])
    plt.title('Top 20 Speakers by Average Word Count per Speech')
    plt.xlabel('Speaker')
    plt.ylabel('Average Word Count per Speech')
    plt.xticks(rotation=22)
    # add the number of speeches in the bar chart
    for i, v in enumerate(speaker_word_count['num_speeches']):
        plt.text(i, v + 5, str(int(v)), ha='center', va='bottom')
    plt.savefig("top 10 speakers by average word count.png")  # Explicitly show the plot (optional, depending on your environment)

def total_word_count(dataset):
    df = dataset.to_pandas()
    # get the total word count
    total_word_count = df['word_count'].sum()
    print(f"Total word count: {total_word_count}")

def graph_contributions_per_speaker(dataset):
    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()

    # Group by speaker and count the number of entries
    speaker_contributions = df.groupby('speaker').size().reset_index(name='counts')
    
    # Sort speakers by contributions in descending order and select top 10
    top_speakers = speaker_contributions.sort_values('counts', ascending=False).head(10)

    # Plot the top 10 speakers
    plt.figure(figsize=(12, 6))
    plt.bar(top_speakers['speaker'], top_speakers['counts'])
    plt.title('Top 10 Speakers by Number of Speeches')
    plt.xlabel('Speaker')
    plt.ylabel('Number of Speeches')
    plt.xticks(rotation=15)
    plt.grid()
    plt.savefig("top 10 speakers")  # Explicitly show the plot (optional, depending on your environment) 

# main function
def main():
    # load dataset
    dataset_dail_en = Dail_dataset(db_fpath="/tmp/debate_db/")
    # read data set from ./nce_ga.txt
    # read the file  
    '''with open('./nce_ga.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        print(text[:1000])
        '''

    # have a look
    '''Dataset({
        features: ['speaker', 'url', 'text', 'date', 'word_count', 'token_count', 'speaker_text'],
        num_rows: 375221
        })'''
    # Visualize the dataset
    # graph the number of words over time
    #graph_word_count_over_time(dataset_dail.dataset)
    #total_word_count(dataset_dail.dataset)
    # graph average word count per speaker
    #top_average_word_count_per_speaker(dataset_dail.dataset)
    # graph by speaker contributions (number of per speaker)
    #graph_contributions_per_speaker(dataset_dail.dataset)


if __name__ == "__main__":
    main()