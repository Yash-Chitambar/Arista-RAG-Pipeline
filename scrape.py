import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
from collections import deque
import re

def create_documents_dir():
    """Create a documents directory if it doesn't exist."""
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' directory")
    else:
        print("'documents' directory already exists")

def download_file(url, save_path):
    """Download a file from a URL and save it to the specified path."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def is_valid_file_extension(url, extensions):
    """Check if the URL has one of the specified file extensions."""
    return any(url.lower().endswith(ext) for ext in extensions)

def is_same_domain(base_url, url):
    """Check if the URL is in the same domain as the base URL."""
    base_domain = urllib.parse.urlparse(base_url).netloc
    url_domain = urllib.parse.urlparse(url).netloc
    
    # Original strict check
    # return base_domain == url_domain
    
    # More lenient check - accept subdomains
    if base_domain == url_domain:
        return True
    
        
    # Check if url_domain is a subdomain of base_domain
    if url_domain.endswith('.' + base_domain):
        return True
        
    # Check if base_domain is a subdomain of url_domain
    if base_domain.endswith('.' + url_domain):
        return True
    
    return False

def clean_url(url):
    """Clean up the URL by removing fragments and normalizing."""
    parsed = urllib.parse.urlparse(url)
    # Remove fragments and normalize
    cleaned = parsed._replace(fragment='')
    # Remove trailing slash for consistency
    path = cleaned.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
        cleaned = cleaned._replace(path=path)
    return urllib.parse.urlunparse(cleaned)

def is_excluded_file(url):
    """Check if the URL points to a file type that should be excluded from crawling."""
    excluded_extensions = [
        # Images
        '.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp', '.svg', '.ico', '.tiff',
        # Downloads
        '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dmg', '.pkg', '.deb', '.rpm',
        # Code files
        '.py', '.java', '.js', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.rs',
        # Other binary files
        '.md', '.db', '.sqlite', '.so', '.dll'
    ]
    return any(url.lower().endswith(ext) for ext in excluded_extensions)

def scrape_page_for_links_and_files(url, base_url, extensions, visited_pages, max_depth):
    """Scrape a page for both links to other pages and file links."""
    print(f"Scraping page: {url}")
    file_links = []
    page_links = []
    
    # Sets to prevent duplicates on the same page
    seen_file_links = set()
    seen_page_links = set()
    
    try:
        # Skip if URL is an excluded file
        if is_excluded_file(url):
            print(f"Skipping excluded file: {url}")
            return [], []
            
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Found {len(soup.find_all('a', href=True))} links on page {url}")
        
        # Find all anchor tags
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute URLs
            absolute_url = urllib.parse.urljoin(url, href)
            cleaned_url = clean_url(absolute_url)
            
            # Skip excluded files for page crawling (but still download if they match extensions)
            if is_excluded_file(cleaned_url) and not is_valid_file_extension(cleaned_url, extensions):
                continue
            
            # Debug URLs being processed
            #if is_valid_file_extension(cleaned_url, extensions):
            #    print(f"Found file: {cleaned_url}")
            
            # Skip already visited pages and external domains
            if cleaned_url in visited_pages:
                #print(f"Skipping already visited: {cleaned_url}")
                continue
            if not is_same_domain(base_url, cleaned_url):
                #print(f"Skipping external domain: {cleaned_url}")
                continue
                
            # Check if it's a file or a page
            if is_valid_file_extension(cleaned_url, extensions):
                if cleaned_url not in seen_file_links:
                    file_links.append(cleaned_url)
                    seen_file_links.add(cleaned_url)
                    print(f"Added file: {cleaned_url}")
                else:
                    #print(f"Skipping duplicate file: {cleaned_url}")
                    pass
            elif cleaned_url.startswith(base_url):  # Only add pages from same domain
                if cleaned_url not in seen_page_links:
                    page_links.append(cleaned_url)
                    seen_page_links.add(cleaned_url)
                    print(f"Added page: {cleaned_url}")
                #else:
                    #print(f"Skipping duplicate page: {cleaned_url}")
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    
    return file_links, page_links

def bfs_crawl(start_url, extensions, max_depth=2, max_pages=30):
    
    """Perform a breadth-first search of the website, prioritizing by distance from root."""
    base_url = urllib.parse.urlparse(start_url).scheme + "://" + urllib.parse.urlparse(start_url).netloc
    
    visited_pages = set()
    file_links = []
    
    # Queue entries are (url, depth)
    queue = deque([(start_url, 0)])
    visited_pages.add(clean_url(start_url))
    
    print(f"Starting BFS crawl from {start_url} with max depth {max_depth}")
    
    while queue and len(visited_pages) < max_pages:
        current_url, current_depth = queue.popleft()
        
        # Skip if we've reached max depth or if it's an excluded file
        if current_depth > max_depth or is_excluded_file(current_url):
            continue
            
        # Process current page
        new_file_links, new_page_links = scrape_page_for_links_and_files(
            current_url, base_url, extensions, visited_pages, max_depth
        )
        
        # Add discovered files
        file_links.extend(new_file_links)
        
        print(f"Found {len(new_file_links)} files on {current_url}")
        
        # If we're not at max depth, add new pages to the queue
        if current_depth < max_depth:
            for page_link in new_page_links:
                cleaned_link = clean_url(page_link)
                # Skip excluded files when adding to the queue
                if cleaned_link not in visited_pages and not is_excluded_file(cleaned_link):
                    queue.append((cleaned_link, current_depth + 1))
                    visited_pages.add(cleaned_link)
    
    # Remove duplicates while preserving order
    unique_file_links = []
    seen = set()
    for link in file_links:
        if link not in seen:
            seen.add(link)
            unique_file_links.append(link)
    
    print(f"BFS crawl complete. Visited {len(visited_pages)} pages, found {len(unique_file_links)} unique files")
    return unique_file_links, visited_pages

def download_files(file_links):
    """Download files from a list of links to the documents directory."""
    if not file_links:
        print("No files to download")
        return
    
    create_documents_dir()
    
    successful_downloads = 0
    for url in file_links:
        # Extract filename from URL
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        
        # If filename is empty or has no extension, generate a name
        if not filename or '.' not in filename:
            filename = f"document_{successful_downloads + 1}{get_extension_from_url(url)}"
        
        save_path = os.path.join("documents", filename)
        
        # Download the file
        if download_file(url, save_path):
            successful_downloads += 1
    
    print(f"\nDownload summary:")
    print(f"Total files found: {len(file_links)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(file_links) - successful_downloads}")

def get_extension_from_url(url):
    """Try to determine file extension from URL."""
    # Common content types and their extensions
    extensions = {
        # Documents
        ".pdf": ".pdf",
        ".docx": ".docx",
        ".doc": ".doc",
        ".txt": ".txt",
        ".rtf": ".rtf",
        ".pages": ".pages",
        ".602": ".602",
        ".abw": ".abw",
        ".cgm": ".cgm",
        ".cwk": ".cwk",
        ".docm": ".docm",
        ".dot": ".dot",
        ".dotm": ".dotm",
        ".hwp": ".hwp",
        ".key": ".key",
        ".lwp": ".lwp",
        ".mw": ".mw",
        ".mcw": ".mcw",
        ".pbd": ".pbd",
        ".wpd": ".wpd",
        ".wps": ".wps",
        ".zabw": ".zabw",
        ".sda": ".sda",
        ".sdd": ".sdd",
        ".sdp": ".sdp",
        ".sdw": ".sdw",
        ".sgl": ".sgl",
        ".sti": ".sti",
        ".sxi": ".sxi",
        ".sxw": ".sxw",
        ".stw": ".stw",
        ".sxg": ".sxg",
        ".vor": ".vor",
        ".xml": ".xml",
        ".epub": ".epub",
        ".uof": ".uof",
        ".uop": ".uop",
        ".uot": ".uot",
        
        # Spreadsheets
        ".csv": ".csv",
        ".xlsx": ".xlsx",
        ".xls": ".xls",
        ".xlsm": ".xlsm",
        ".xlsb": ".xlsb",
        ".xlw": ".xlw",
        ".dif": ".dif",
        ".sylk": ".sylk",
        ".slk": ".slk",
        ".prn": ".prn",
        ".numbers": ".numbers",
        ".et": ".et",
        ".ods": ".ods",
        ".fods": ".fods",
        ".uos1": ".uos1",
        ".uos2": ".uos2",
        ".dbf": ".dbf",
        ".wk1": ".wk1",
        ".wk2": ".wk2",
        ".wk3": ".wk3",
        ".wk4": ".wk4",
        ".wks": ".wks",
        ".123": ".123",
        ".wq1": ".wq1",
        ".wq2": ".wq2",
        ".wb1": ".wb1",
        ".wb2": ".wb2",
        ".wb3": ".wb3",
        ".qpw": ".qpw",
        ".xlr": ".xlr",
        ".eth": ".eth",
        ".tsv": ".tsv",
        
        # Presentations
        ".ppt": ".ppt",
        ".pptx": ".pptx",
        ".pptm": ".pptm",
        ".pot": ".pot",
        ".potm": ".potm",
        ".potx": ".potx",
        
        # Other formats
        ".json": ".json",
        ".html": ".html",
        ".htm": ".htm"
    }
    
    for ext in extensions:
        if url.lower().endswith(ext):
            return ext
    
    # Default extension if we can't determine
    return ".html"

def scrape_and_download(start_url, max_pages = float('inf')):
    """Main function to scrape and download files."""
    # Target URL to scrape - set to Berkeley site


    
    # Maximum depth for BFS crawl
    max_depth = 3
    
    # File extensions to look for
    valid_extensions = [
        # Documents
        '.pdf', '.docx', '.doc', '.txt', '.rtf', '.pages', '.602', '.abw', '.cgm', '.cwk', 
        '.docm', '.dot', '.dotm', '.hwp', '.key', '.lwp', '.mw', '.mcw', '.pbd', '.wpd', 
        '.wps', '.zabw', '.sda', '.sdd', '.sdp', '.sdw', '.sgl', '.sti', '.sxi', '.sxw', 
        '.stw', '.sxg', '.vor', '.xml', '.epub', '.uof', '.uop', '.uot',
        
        # Spreadsheets
        '.csv', '.xlsx', '.xls', '.xlsm', '.xlsb', '.xlw', '.dif', '.sylk', '.slk', '.prn',
        '.numbers', '.et', '.ods', '.fods', '.uos1', '.uos2', '.dbf', '.wk1', '.wk2', '.wk3',
        '.wk4', '.wks', '.123', '.wq1', '.wq2', '.wb1', '.wb2', '.wb3', '.qpw', '.xlr', '.eth', '.tsv',
        
        # Presentations
        '.ppt', '.pptx', '.pptm', '.pot', '.potm', '.potx',
        
    ]
    

    
    # Perform BFS crawl to find files
    file_links, visited_pages = bfs_crawl(start_url, valid_extensions, max_depth, max_pages)
    
    print(f"Total visited pages: {len(visited_pages)}")
    
    # Download the files
    download_files(file_links)

if __name__ == "__main__":

    arista = "https://www.arista.com/en/"
    cs61a = "https://cs61a.org"

    scrape_and_download(arista, 1000)
    
     