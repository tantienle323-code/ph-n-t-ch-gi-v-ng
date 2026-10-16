import re
from datetime import date, datetime, timedelta, timezone
from html import unescape
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

CRISIS_KEYWORDS = {
    'high_risk': ['war', 'missile', 'attack', 'invasion', 'conflict', 'sanctions', 'escalation', 'nuclear', 'bomb'],
    'medium_risk': ['crisis', 'inflation', 'fed rate', 'recession', 'tensions', 'tariff', 'yield spike'],
    'positive': ['peace', 'ceasefire', 'deal', 'agreement', 'growth', 'recovery', 'soft landing']
}

GOOGLE_NEWS_RSS_URL = 'https://news.google.com/rss/search'
MYMEMORY_TRANSLATE_URL = 'https://api.mymemory.translated.net/get'
MAX_NEWS_ARTICLES = 10
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
MONTH_NAMES = (
    'January|February|March|April|May|June|July|August|September|October|November|December|'
    'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec'
)
HEADLINE_DATE_PATTERNS = (
    re.compile(rf'\b({MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b', re.IGNORECASE),
    re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
)


class NewsFetchError(RuntimeError):
    """A safe, user-facing error for unavailable or invalid RSS responses."""


class NewsTranslationError(RuntimeError):
    """A safe, user-facing error for unavailable or invalid translations."""


def translate_headline_to_vietnamese(text):
    """Translate an English headline to Vietnamese using MyMemory."""
    try:
        response = requests.get(
            MYMEMORY_TRANSLATE_URL,
            params={'q': text, 'langpair': 'en|vi'},
            timeout=(5, 15),
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        raise NewsTranslationError('Không thể dịch tiêu đề tin tức lúc này.') from None

    if not isinstance(result, dict) or result.get('responseStatus') != 200:
        raise NewsTranslationError('Dịch vụ dịch không trả về kết quả hợp lệ.')

    response_data = result.get('responseData')
    translated_text = (
        response_data.get('translatedText', '')
        if isinstance(response_data, dict) else ''
    )
    if not isinstance(translated_text, str) or not translated_text.strip():
        raise NewsTranslationError('Dịch vụ dịch không trả về kết quả hợp lệ.')
    return unescape(translated_text).strip()


def analyze_headline_sentiment(text):
    """Analyze a headline for sentiment and geopolitical risk keywords."""
    text_lower = text.lower()

    risk_score = 0
    crisis_flag = 0

    for keyword in CRISIS_KEYWORDS['high_risk']:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            risk_score += 2.0
            crisis_flag = 1

    for keyword in CRISIS_KEYWORDS['medium_risk']:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            risk_score += 1.0

    positive_score = sum(
        1.0 for keyword in CRISIS_KEYWORDS['positive']
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower)
    )

    return positive_score - risk_score, crisis_flag


def _explicit_headline_dates(title):
    for match in HEADLINE_DATE_PATTERNS[0].finditer(title):
        month, day, year = match.groups()
        try:
            yield datetime.strptime(f'{month} {day} {year}', '%B %d %Y').date()
        except ValueError:
            try:
                yield datetime.strptime(f'{month} {day} {year}', '%b %d %Y').date()
            except ValueError:
                continue

    for month_or_day, day_or_month, year in HEADLINE_DATE_PATTERNS[1].findall(title):
        for month, day in ((day_or_month, month_or_day), (month_or_day, day_or_month)):
            try:
                yield date(int(year), int(month), int(day))
                break
            except ValueError:
                continue

    for year, month, day in HEADLINE_DATE_PATTERNS[2].findall(title):
        try:
            yield date(int(year), int(month), int(day))
        except ValueError:
            continue


def parse_google_news_rss(xml_content, reference_date=None):
    """Parse Google News RSS XML into scored headlines."""
    root = ElementTree.fromstring(xml_content)
    news_details = []
    reference_date = reference_date or datetime.now(VIETNAM_TIMEZONE).date()

    for item in root.findall('./channel/item'):
        title = (item.findtext('title') or '').strip()
        if not title:
            continue
        if any(headline_date > reference_date for headline_date in _explicit_headline_dates(title)):
            continue

        source = (item.findtext('source') or '').strip()
        article_url = (item.findtext('link') or '').strip()
        if urlparse(article_url).scheme not in {'http', 'https'}:
            article_url = ''
        published_text = (item.findtext('pubDate') or '').strip()
        published_at = 'Unknown'
        if published_text:
            published_datetime = parsedate_to_datetime(published_text).astimezone(
                VIETNAM_TIMEZONE
            )
            published_at = published_datetime.strftime('%Y-%m-%d %H:%M ICT (UTC+7)')
        sentiment, crisis_flag = analyze_headline_sentiment(title)
        news_details.append({
            'title': title,
            'source': source or 'Google News',
            'url': article_url,
            'published_at': published_at,
            'sentiment': sentiment,
            'is_crisis': bool(crisis_flag),
        })

    if not news_details:
        raise ValueError('Google News RSS returned no usable headlines.')

    news_details.sort(
        key=lambda article: article['published_at'] if article['published_at'] != 'Unknown' else '',
        reverse=True,
    )
    news_details = news_details[:MAX_NEWS_ARTICLES]
    total_sentiment = sum(item['sentiment'] for item in news_details)
    crisis_count = sum(item['is_crisis'] for item in news_details)
    return {
        'avg_sentiment': total_sentiment / len(news_details),
        'crisis_flag': int(crisis_count > 0),
        'crisis_count': crisis_count,
        'news_list': news_details,
        'source': 'Google News RSS',
    }


def get_latest_news_sentiment():
    """Fetch current gold and macroeconomic headlines from Google News RSS."""
    try:
        response = requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={
                'q': 'gold price OR gold market OR Federal Reserve OR inflation',
                'hl': 'en-US',
                'gl': 'US',
                'ceid': 'US:en',
            },
            timeout=(5, 15),
        )
        response.raise_for_status()
        return parse_google_news_rss(response.content)
    except requests.RequestException:
        raise NewsFetchError('Không thể kết nối Google News RSS. Vui lòng thử lại sau.') from None
    except (ElementTree.ParseError, ValueError):
        raise NewsFetchError('Google News RSS trả về dữ liệu không hợp lệ hoặc không có tiêu đề.') from None


if __name__ == '__main__':
    result = get_latest_news_sentiment()
    print(f"Source: {result['source']}")
    print(f"Average headline sentiment: {result['avg_sentiment']:.2f}")
    print(f"Headlines containing high-risk terms: {result['crisis_count']}/{len(result['news_list'])}")
    for article in result['news_list']:
        print(f"{article['published_at']} | {article['source']} | {article['title']}")
