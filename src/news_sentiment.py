import os
import re
import pandas as pd
import numpy as np

CRISIS_KEYWORDS = {
    'high_risk': ['war', 'missile', 'attack', 'invasion', 'conflict', 'sanctions', 'escalation', 'nuclear', 'bomb'],
    'medium_risk': ['crisis', 'inflation', 'fed rate', 'recession', 'tensions', 'tariff', 'yield spike'],
    'positive': ['peace', 'ceasefire', 'deal', 'agreement', 'growth', 'recovery', 'soft landing']
}

def analyze_headline_sentiment(text):
    """Phân tích cảm xúc và phát hiện rủi ro địa chính trị từ tiêu đề tin tức."""
    text_lower = text.lower()
    
    risk_score = 0
    crisis_flag = 0
    
    for kw in CRISIS_KEYWORDS['high_risk']:
        if re.search(r'\b' + kw + r'\b', text_lower):
            risk_score += 2.0
            crisis_flag = 1
            
    for kw in CRISIS_KEYWORDS['medium_risk']:
        if re.search(r'\b' + kw + r'\b', text_lower):
            risk_score += 1.0
            
    pos_score = 0
    for kw in CRISIS_KEYWORDS['positive']:
        if re.search(r'\b' + kw + r'\b', text_lower):
            pos_score += 1.0
            
    sentiment_score = pos_score - risk_score
    return sentiment_score, crisis_flag

def get_latest_news_sentiment():
    """Giả lập & Quét dòng tin tức tài chính ngắn hạn về Vàng & Địa chính trị."""
    sample_news = [
        {"title": "Middle East tensions escalate as missile strikes reported near oil facilities", "source": "Reuters"},
        {"title": "Fed signals cautious approach on interest rate cuts amid sticky inflation", "source": "Bloomberg"},
        {"title": "Gold rallies to record highs as geopolitical crisis drives safe-haven demand", "source": "MarketWatch"},
        {"title": "Central banks continue aggressive gold buying to hedge against currency sanctions", "source": "FT"},
        {"title": "US dollar strengthens slightly as Treasury yield holds above 4.5%", "source": "CNBC"}
    ]
    
    total_sentiment = 0.0
    total_crisis = 0
    news_details = []
    
    for news in sample_news:
        sent, crisis = analyze_headline_sentiment(news['title'])
        total_sentiment += sent
        if crisis:
            total_crisis += 1
        news_details.append({
            'title': news['title'],
            'source': news['source'],
            'sentiment': sent,
            'is_crisis': bool(crisis)
        })
        
    avg_sentiment = total_sentiment / len(sample_news)
    overall_crisis = 1 if total_crisis > 0 else 0
    
    return {
        'avg_sentiment': avg_sentiment,
        'crisis_flag': overall_crisis,
        'crisis_count': total_crisis,
        'news_list': news_details
    }

if __name__ == '__main__':
    result = get_latest_news_sentiment()
    print("==================================================")
    print("  RADA PHÂN TÍCH TÍN HIỆU TIN TỨC ĐỊA CHÍNH TRỊ ")
    print("==================================================")
    print(f"• Điểm cảm xúc tin tức trung bình: {result['avg_sentiment']:.2f}")
    print(f"• Cảnh báo Xung đột Địa chính trị: {'🔴 RỦI RO CAO' if result['crisis_flag'] else '🟢 BÌNH THƯỜNG'}")
    print(f"• Số tin tức chứa từ khóa rủi ro: {result['crisis_count']}/{len(result['news_list'])}")