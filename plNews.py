from newsapi import NewsApiClient
from sentiment import estimate_sentiment

newsapi = NewsApiClient(api_key='a837300c7ef24100afa35940642798b4')

def get_news_sentiment():
    try:
        all_articles = newsapi.get_everything(
            q='(forex OR trading OR finance)', 
            language='en',
            sort_by='publishedAt',
            page=1,
            page_size=100
        )
        
        print(f"Status: {all_articles['status']}")
        print(f"Total Results: {all_articles['totalResults']}\n")

        if all_articles['articles']:
            news = [article['title'] for article in all_articles['articles']]
            
            if not news:
                print(f'No news found for sentiment analysis')
                probability, sentiment = 0, "neutral"
            else:
                probability, sentiment = estimate_sentiment(news)
                
            print(f"Overall Sentiment: {sentiment}")
            print(f"Confidence: {probability:.2%}")
            
            return probability, sentiment
        else:
            print("No articles found!")

        return None, None

    except Exception as e:
        print(f"An error occurred: {str(e)}")

