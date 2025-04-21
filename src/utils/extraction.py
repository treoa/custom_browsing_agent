"""Content Extraction Utilities

This module provides utilities for extracting content from web pages
and various text sources to enhance the research agent's information gathering.
"""

import re
import json
import html
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse

from bs4 import BeautifulSoup


class ContentExtractor:
    """
    Utility class for extracting and processing content from various sources.
    
    This class provides methods to extract clean, structured content from
    web pages, HTML, and other text sources to support research tasks.
    """
    
    @staticmethod
    def extract_main_content(html_content: str) -> str:
        """
        Extract the main content from an HTML page, removing navigation, ads, etc.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Extracted main content as text
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove common non-content elements
            for element in soup.select('nav, header, footer, aside, script, style, iframe, [class*="nav"], [class*="header"], [class*="footer"], [class*="menu"], [class*="banner"], [id*="nav"], [id*="header"], [id*="footer"], [id*="menu"], [id*="banner"]'):
                element.decompose()
            
            # Try to find main content area
            main_content = None
            
            # Common content containers
            content_selectors = [
                'main', 'article', '[role="main"]', '#content', '.content', '#main', '.main',
                '[class*="content"]', '[class*="article"]', '[id*="content"]', '[id*="article"]'
            ]
            
            # Try each selector until we find content
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Find the largest element by content length
                    main_content = max(elements, key=lambda x: len(x.get_text(strip=True)))
                    break
            
            # If no content container found, use body
            if not main_content:
                main_content = soup.body or soup
            
            # Extract text with some formatting
            text = ''
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 'blockquote']):
                content = element.get_text(strip=True)
                if content:
                    if element.name.startswith('h'):
                        level = int(element.name[1])
                        text += '\n' + '#' * level + ' ' + content + '\n'
                    elif element.name == 'blockquote':
                        text += '\n> ' + content + '\n'
                    elif element.name == 'li':
                        text += '\n- ' + content
                    else:
                        text += '\n' + content + '\n'
            
            # Clean up extra whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        except Exception:
            # Fallback to basic text extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
    
    @staticmethod
    def extract_article_metadata(html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract metadata about an article (title, author, date, etc.)
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "title": "",
            "author": "",
            "date": "",
            "source": urlparse(url).netloc,
            "url": url
        }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('meta', property='og:title') or \
                      soup.find('meta', attrs={'name': 'twitter:title'}) or \
                      soup.find('title')
            
            if title_tag:
                if title_tag.get('content'):
                    metadata["title"] = title_tag.get('content')
                else:
                    metadata["title"] = title_tag.string
            
            # Extract author
            author_tag = soup.find('meta', property='article:author') or \
                       soup.find('meta', attrs={'name': 'author'}) or \
                       soup.find('a', attrs={'rel': 'author'})
            
            if author_tag:
                if author_tag.get('content'):
                    metadata["author"] = author_tag.get('content')
                else:
                    metadata["author"] = author_tag.string
            
            # Extract date
            date_tag = soup.find('meta', property='article:published_time') or \
                     soup.find('meta', attrs={'name': 'date'}) or \
                     soup.find('time')
            
            if date_tag:
                if date_tag.get('content'):
                    metadata["date"] = date_tag.get('content')
                elif date_tag.get('datetime'):
                    metadata["date"] = date_tag.get('datetime')
                else:
                    metadata["date"] = date_tag.string
        
        except Exception:
            # Fallback: return with partial metadata
            pass
            
        return metadata
    
    @staticmethod
    def extract_links(html_content: str, base_url: str) -> List[Dict[str, str]]:
        """
        Extract links from HTML content with context.
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link dictionaries with url, text, and context
        """
        links = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '')
                
                # Skip empty, javascript, or anchor links
                if not href or href.startswith(('javascript:', '#', 'mailto:')):
                    continue
                
                # Get link text
                link_text = a_tag.get_text(strip=True)
                
                # Get surrounding context
                parent = a_tag.parent
                context = parent.get_text(strip=True) if parent else ""
                
                # Limit context length
                if len(context) > 300:
                    context = context[:300] + "..."
                
                # Get title attribute or alt text of contained images
                title = a_tag.get('title', '')
                if not title:
                    img = a_tag.find('img')
                    if img and img.get('alt'):
                        title = img.get('alt', '')
                
                # Create link info
                link_info = {
                    "url": href,
                    "text": link_text,
                    "title": title,
                    "context": context
                }
                
                links.append(link_info)
        
        except Exception:
            # Fallback: return empty list
            pass
            
        return links
    
    @staticmethod
    def extract_structured_data(html_content: str) -> List[Dict[str, Any]]:
        """
        Extract structured JSON-LD data from HTML content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of structured data objects
        """
        structured_data = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all JSON-LD script tags
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    # Parse JSON content
                    data = json.loads(script.string)
                    structured_data.append(data)
                except Exception:
                    continue
        
        except Exception:
            # Fallback: return empty list
            pass
            
        return structured_data
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace, HTML entities, etc.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_tables(html_content: str) -> List[List[List[str]]]:
        """
        Extract tables from HTML content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cells
        """
        tables = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for table_tag in soup.find_all('table'):
                table_data = []
                
                # Extract headers
                headers = []
                header_row = table_tag.find('thead')
                if header_row:
                    for th in header_row.find_all('th'):
                        headers.append(th.get_text(strip=True))
                    
                    if headers:
                        table_data.append(headers)
                
                # Extract rows
                for tr in table_tag.find_all('tr'):
                    row = []
                    
                    # Extract cells (td or th)
                    cells = tr.find_all(['td', 'th'])
                    if cells:
                        for cell in cells:
                            row.append(cell.get_text(strip=True))
                        
                        table_data.append(row)
                
                # Add table if it has data
                if table_data and len(table_data) > 1:  # More than just headers
                    tables.append(table_data)
        
        except Exception:
            # Fallback: return empty list
            pass
            
        return tables
    
    @staticmethod
    def extract_key_points(text: str, max_points: int = 10) -> List[str]:
        """
        Extract key points or sentences from a text.
        
        Args:
            text: Source text
            max_points: Maximum number of key points to extract
            
        Returns:
            List of key points
        """
        # Simple extraction based on sentence characteristics
        # In a real system, would use NLP for better extraction
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out too short sentences and normalize
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip too short sentences or incomplete ones
            if len(sentence) > 15 and sentence[-1] in ['.', '!', '?']:
                valid_sentences.append(sentence)
        
        # Basic heuristics for importance (length, keywords, position)
        scored_sentences = []
        important_markers = ['key', 'important', 'significant', 'critical', 'essential', 
                            'crucial', 'fundamental', 'central', 'primary', 'vital',
                            'notably', 'specifically', 'particularly', 'especially']
        
        for i, sentence in enumerate(valid_sentences):
            score = 0
            
            # Position score - earlier and later sentences often more important
            if i < len(valid_sentences) * 0.2 or i > len(valid_sentences) * 0.8:
                score += 2
            
            # Length score - not too short, not too long
            length = len(sentence)
            if 30 < length < 200:
                score += 1
            
            # Keyword score
            lower_s = sentence.lower()
            for marker in important_markers:
                if marker in lower_s:
                    score += 2
                    break
            
            # Title-like score (if capitalized like a title)
            words = sentence.split()
            if len(words) > 3 and len(words) < 12:
                if all(w[0].isupper() for w in words if len(w) > 3):
                    score += 3
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top N
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored_sentences[:max_points]]
