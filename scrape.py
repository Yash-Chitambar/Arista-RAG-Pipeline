import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import urllib.parse
from collections import deque
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

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
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        print(f"Downloaded: {save_path} from {url}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

@lru_cache(maxsize=1000)
def is_valid_file_extension(url, extensions_tuple):
    """Check if the URL has one of the specified file extensions."""
    extensions = set(extensions_tuple)  # Convert tuple back to set
    return any(url.lower().endswith(ext) for ext in extensions)

def is_same_domain(base_url, url):
    """Check if the URL is in the same domain as the base URL."""
    base_domain = urllib.parse.urlparse(base_url).netloc
    url_domain = urllib.parse.urlparse(url).netloc
    
    if base_domain == url_domain:
        return True
    
    if url_domain.endswith('.' + base_domain):
        return True
        
    if base_domain.endswith('.' + url_domain):
        return True
    
    return False

@lru_cache(maxsize=1000)
def clean_url(url):
    """Clean up the URL by removing fragments and normalizing."""
    parsed = urllib.parse.urlparse(url)
    cleaned = parsed._replace(fragment='')
    path = cleaned.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
        cleaned = cleaned._replace(path=path)
    return urllib.parse.urlunparse(cleaned)

@lru_cache(maxsize=1000)
def is_excluded_file(url):
    """Check if the URL points to a file type that should be excluded from crawling."""
    excluded_extensions = (
        # Images
        '.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp', '.svg', '.ico', '.tiff',
        # Downloads
        '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dmg', '.pkg', '.deb', '.rpm',
        # Code files
        '.py', '.java', '.js', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.rs',
        # Other binary files
        '.md', '.db', '.sqlite', '.so', '.dll'
    )
    return any(url.lower().endswith(ext) for ext in excluded_extensions)

def scrape_page_for_links_and_files(url, base_url, extensions, visited_pages, max_depth):
    """Scrape a page for both links to other pages and file links using Selenium."""
    print(f"Scraping page: {url}")
    file_links = set()
    page_links = set()
    
    try:
        # Initialize Chrome WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        
        # Set page load timeout
        driver.set_page_load_timeout(30)
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Find all anchor tags
        links = driver.find_elements(By.TAG_NAME, "a")
        print(f"Found {len(links)} links on page {url}")
        
        for link in links:
            try:
                href = link.get_attribute("href")
                if not href:
                    continue
                    
                # Convert relative URLs to absolute URLs
                absolute_url = urllib.parse.urljoin(url, href)
                cleaned_url = clean_url(absolute_url)
                
                # Skip excluded files for page crawling
                if is_excluded_file(cleaned_url) and not is_valid_file_extension(cleaned_url, extensions):
                    continue
                
                # Skip already visited pages and external domains
                if cleaned_url in visited_pages:
                    continue
                if not is_same_domain(base_url, cleaned_url):
                    continue
                
                # Check if it's a file or a page
                if is_valid_file_extension(cleaned_url, extensions):
                    file_links.add(cleaned_url)
                    print(f"Added file: {cleaned_url}")
                elif cleaned_url.startswith(base_url):
                    page_links.add(cleaned_url)
                    print(f"Added page: {cleaned_url}")
                    
            except Exception as e:
                print(f"Error processing link: {e}")
                continue
                
    except TimeoutException:
        print(f"Timeout while loading {url}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass
    
    return list(file_links), list(page_links)

def bfs_crawl(start_url, extensions, max_depth=2, max_pages=30, max_files=float('inf')):
    """Perform a breadth-first search of the website, prioritizing by distance from root."""
    base_url = urllib.parse.urlparse(start_url).scheme + "://" + urllib.parse.urlparse(start_url).netloc
    
    visited_pages = set()
    file_links = set()
    extensions_tuple = tuple(extensions)
    
    queue = deque([(start_url, 0)])
    visited_pages.add(clean_url(start_url))
    
    print(f"Starting BFS crawl from {start_url} with max depth {max_depth}")
    start_time = time.time()
    
    while queue and len(visited_pages) < max_pages and len(file_links) < max_files:
        current_url, current_depth = queue.popleft()
        
        if current_depth > max_depth or is_excluded_file(current_url):
            continue
            
        new_file_links, new_page_links = scrape_page_for_links_and_files(
            current_url, base_url, extensions_tuple, visited_pages, max_depth
        )
        
        file_links.update(new_file_links)
        
        if len(file_links) >= max_files:
            print(f"Reached maximum number of files ({max_files})")
            break
        
        if current_depth < max_depth:
            for page_link in new_page_links:
                cleaned_link = clean_url(page_link)
                if cleaned_link not in visited_pages and not is_excluded_file(cleaned_link):
                    queue.append((cleaned_link, current_depth + 1))
                    visited_pages.add(cleaned_link)
    
    end_time = time.time()
    print(f"BFS crawl complete. Visited {len(visited_pages)} pages, found {len(file_links)} unique files")
    print(f"Crawl time: {end_time - start_time:.2f} seconds")
    return list(file_links), visited_pages

def download_files(file_links):
    """Download files from a list of links to the documents directory using parallel processing."""
    if not file_links:
        print("No files to download")
        return
    
    create_documents_dir()
    
    successful_downloads = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list of futures
        future_to_url = {}
        for url in file_links:
            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if not filename or '.' not in filename:
                filename = f"document_{successful_downloads + 1}{get_extension_from_url(url)}"
            save_path = os.path.join("documents", filename)
            future = executor.submit(download_file, url, save_path)
            future_to_url[future] = url
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                if future.result():
                    successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    
    end_time = time.time()
    print(f"\nDownload summary:")
    print(f"Total files found: {len(file_links)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(file_links) - successful_downloads}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

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

def scrape_and_download(start_url, max_depth = float('inf'), max_pages = float('inf'), max_files = float('inf')):
    """Main function to scrape and download files."""
    # Target URL to scrape - set to Berkeley site


    
    # Maximum depth for BFS crawl
    
    
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
    file_links, visited_pages = bfs_crawl(start_url, valid_extensions, max_depth, max_pages, max_files)
    
    # Download the files
    download_files(file_links)
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total pages visited: {len(visited_pages)}")
    print(f"Total files found: {len(file_links)}")
    print(f"Download directory size: {sum(os.path.getsize(os.path.join('documents', f)) for f in os.listdir('documents') if os.path.isfile(os.path.join('documents', f))) / (1024*1024):.2f} MB")

if __name__ == "__main__":

    arista = "https://www.arista.com/en/"
    cs61a = "https://cs61a.org"

    #leave last two parameters blank to scrape indefinitely
    scrape_and_download(cs61a,max_depth = 10, max_pages=10000, max_files=10)
    
     