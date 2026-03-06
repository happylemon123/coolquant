import urllib.request
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinancialSentimentAnalysis:
    """
    Uses FinBERT (ProsusAI/finbert) to generate a numeric sentiment signal
    from a ticker's latest news.
    """
    def __init__(self):
        print("FinBERT: Loading tokenizer and model (this might take a few seconds)...")
        # Load pre-trained model and tokenizer for financial sentiment
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
    def fetch_recent_news(self, ticker: str, max_articles: int = 5) -> list[str]:
        """
        Fetches the latest news headlines for a given ticker using the Yahoo Finance API.
        """
        try:
            url = f'https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount={max_articles}'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            
            headlines = []
            if 'news' in data:
                for item in data['news'][:max_articles]:
                    title = item.get('title', '')
                    if title:
                        headlines.append(title)
            return headlines
        except Exception as e:
            print(f"Warning: Failed to fetch news for {ticker}: {e}")
            return []

    def analyze_sentiment(self, text: str) -> float:
        """
        Calculates sentiment using FinBERT.
        Returns a score from -1.0 (Bearish) to 1.0 (Bullish).
        FinBERT outputs logits for [positive, negative, neutral].
        """
        if not text:
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # ProsusAI/finbert mapping: 0=positive, 1=negative, 2=neutral
        pos_prob = probs[0][0].item()
        neg_prob = probs[0][1].item()
        
        # Calculate a net sentiment score (positive prob - negative prob)
        score = pos_prob - neg_prob
        return score

    def get_ticker_sentiment_signal(self, ticker: str, max_articles: int = 5) -> float:
        """
        Fetches recent news and returns an aggregated sentiment score.
        """
        headlines = self.fetch_recent_news(ticker, max_articles)
        
        if not headlines:
            print(f"No news found for {ticker}, returning neutral sentiment (0.0).")
            return 0.0
            
        scores = []
        for headline in headlines:
            score = self.analyze_sentiment(headline)
            scores.append(score)
            
        # Calculate average sentiment across recent headlines
        avg_score = np.mean(scores)
        return float(avg_score)

if __name__ == "__main__":
    # Test execution
    analyzer = FinancialSentimentAnalysis()
    
    ticker = "AAPL"
    print(f"\nFetching news for {ticker}...")
    headlines = analyzer.fetch_recent_news(ticker)
    
    print("\nIndividual Headline Sentiments:")
    for headline in headlines:
        score = analyzer.analyze_sentiment(headline)
        print(f"[{score:+.2f}] {headline}")
        
    final_signal = analyzer.get_ticker_sentiment_signal(ticker)
    print(f"\nFinal Aggregated Sentiment Alpha Signal for {ticker}: {final_signal:+.4f}")
