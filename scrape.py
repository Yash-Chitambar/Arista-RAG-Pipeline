import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
from getpass import getpass

def create_documents_dir():
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' directory")
    else:
        print("'documents' directory already exists")

def download_file(url, save_path):
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
    return any(url.lower().endswith(ext) for ext in extensions_tuple)

def is_same_domain(base_url, url):
    base_domain = urllib.parse.urlparse(base_url).netloc
    url_domain = urllib.parse.urlparse(url).netloc
    return (base_domain == url_domain or 
            url_domain.endswith('.' + base_domain) or 
            base_domain.endswith('.' + url_domain))

@lru_cache(maxsize=1000)
def clean_url(url):
    parsed = urllib.parse.urlparse(url)
    cleaned = parsed._replace(fragment='')
    path = cleaned.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
        cleaned = cleaned._replace(path=path)
    return urllib.parse.urlunparse(cleaned)

@lru_cache(maxsize=1000)
def is_excluded_file(url):
    excluded_extensions = (
        '.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp', '.svg', '.ico', '.tiff',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dmg', '.pkg', '.deb', '.rpm',
        '.py', '.java', '.js', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.rs',
        '.md', '.db', '.sqlite', '.so', '.dll'
    )
    return any(url.lower().endswith(ext) for ext in excluded_extensions)

def scrape_page_for_links_and_files(url, base_url, extensions, old_visited_pages, old_file_links, driver):
    print(f"Scraping page: {url}")
    page_links = set()
    file_links = set()
    try:
        driver.set_page_load_timeout(30)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        links = driver.find_elements(By.TAG_NAME, "a")

        for link in links:
            try:
                href = link.get_attribute("href")
                if not href:
                    continue
                absolute_url = urllib.parse.urljoin(url, href)
                cleaned_url = clean_url(absolute_url)
                if is_excluded_file(cleaned_url) and not is_valid_file_extension(cleaned_url, extensions):
                    continue
                if cleaned_url in old_visited_pages or cleaned_url in old_file_links:
                    continue
                if not is_same_domain(base_url, cleaned_url):
                    continue
                if is_valid_file_extension(cleaned_url, extensions):
                    file_links.add(cleaned_url)
                    print(f"Added file: {cleaned_url}")
                elif cleaned_url.startswith(base_url):
                    page_links.add(cleaned_url)
                    print(f"Added page: {cleaned_url}")
            except Exception as e:
                print(f"Error processing link: {e}")
    except TimeoutException:
        print(f"Timeout while loading {url}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    print(f"Found {len(file_links)} files and {len(page_links)} new pages on page {url}")
    return file_links, page_links

def bfs_crawl(start_url, extensions, max_pages, max_files, driver):
    base_url = urllib.parse.urlparse(start_url).scheme + "://" + urllib.parse.urlparse(start_url).netloc
    visited_pages = set()
    file_links = set()
    extensions_tuple = tuple(extensions)
    pages_scraped = 0
    queue = deque([(start_url, 0)])
    visited_pages.add(clean_url(start_url))

    print(f"Starting BFS crawl from {start_url}")
    start_time = time.time()

    try:
        while queue and pages_scraped < max_pages and len(file_links) < max_files:
            current_url, current_depth = queue.popleft()
            if is_excluded_file(current_url):
                continue
            new_file_links, new_page_links = scrape_page_for_links_and_files(
                current_url, base_url, extensions_tuple, visited_pages, file_links, driver
            )
            file_links.update(new_file_links)
            if len(file_links) >= max_files:
                print(f"Reached maximum number of files ({max_files})")
                break
            for page_link in new_page_links:
                cleaned_link = clean_url(page_link)
                if cleaned_link not in visited_pages and not is_excluded_file(cleaned_link):
                    queue.append((cleaned_link, current_depth + 1))
                    visited_pages.add(cleaned_link)
            pages_scraped += 1
    except Exception as e:
        print(f"Error during BFS crawl: {e}")
    end_time = time.time()
    print(f"BFS crawl complete. Scraped {pages_scraped} pages, found {len(file_links)} unique files")
    print(f"Crawl time: {end_time - start_time:.2f} seconds")
    return file_links, pages_scraped

def download_files(file_links, max_files):
    if not file_links:
        print("No files to download")
        return
    create_documents_dir()
    successful_downloads = 0
    count = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {}
        for url in file_links:
            if count >= max_files:
                break
            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if not filename or '.' not in filename:
                filename = f"document_{successful_downloads + 1}{get_extension_from_url(url)}"
            save_path = os.path.join("documents", filename)
            future = executor.submit(download_file, url, save_path)
            future_to_url[future] = url
            count += 1
        for future in as_completed(future_to_url):
            try:
                if future.result():
                    successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {future_to_url[future]}: {e}")
    end_time = time.time()
    print(f"\nDownload summary:")
    print(f"Total files found: {len(file_links)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(file_links) - successful_downloads}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

def get_extension_from_url(url):
    common_extensions = ['.pdf', '.docx', '.xlsx', '.csv', '.txt', '.pptx']
    for ext in common_extensions:
        if url.lower().endswith(ext):
            return ext
    return ".html"

def login_to_website(driver, base_url):
    login_url = f"{base_url}/login"
    print(f"Attempting to login at {login_url}")
    try:
        driver.get(login_url)

        # Email input
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "username")))
        username = input("Please enter your email: ")
        email_input = driver.find_element(By.ID, "username")
        email_input.clear()
        email_input.send_keys(username)
        driver.find_element(By.ID, "btnLoginOrigin").click()

        # Password input - now using getpass for secure input
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "password")))
        password = getpass("Please enter your password: ")
        password_input = driver.find_element(By.ID, "password")
        password_input.clear()
        password_input.send_keys(password)
        driver.find_element(By.ID, "btnLogin").click()

        # OTP input
        otp = input("Please enter the 6-digit OTP sent to your email: ").strip()
        if len(otp) != 6 or not otp.isdigit():
            print("Invalid OTP format. It must be exactly 6 digits.")
            return False

        for i in range(6):
            otp_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, f"otp2fa_t{i+1}"))
            )
            otp_field.clear()
            otp_field.send_keys(otp[i])
            time.sleep(0.1)

        # Handle popup if blocking the button
        try:
            close_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.alert-success button.close, div.alert-success .close"))
            )
            close_btn.click()
            print("Closed success popup manually.")
        except Exception:
            print("No close button on success popup. Waiting for it to disappear...")

        try:
            WebDriverWait(driver, 10).until(
                EC.invisibility_of_element_located((By.CLASS_NAME, "alert-success"))
            )
            print("Popup is now gone.")
        except:
            print("Popup did not disappear. Proceeding with force click.")

        # Scroll and click OTP submit button via JS
        try:
            button = driver.find_element(By.ID, "btnLoginOtp")
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", button)
            print("OTP submitted via JavaScript.")
        except Exception as e:
            print(f"Failed to click OTP button: {e}")
            return False

        print("Successfully logged in!")
        return True

    except Exception as e:
        print(f"Login failed: {e}")
        return False

def scrape_and_download(start_url, max_pages=float('inf'), max_files=float('inf')):
    valid_extensions = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv', '.ppt', '.pptx']
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    if not login_to_website(driver, start_url):
        driver.quit()
        return

    file_links, pages_scraped = bfs_crawl(start_url, valid_extensions, max_pages, max_files, driver)
    download_files(file_links, max_files)
    driver.quit()

    print("\nFinal Statistics:")
    print(f"Total pages scraped: {pages_scraped}")
    print(f"Total files found: {len(file_links)}")
    print(f"Download directory size: {sum(os.path.getsize(os.path.join('documents', f)) for f in os.listdir('documents') if os.path.isfile(os.path.join('documents', f))) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    start_url = "https://www.arista.com/en"

    scrape_and_download(start_url, max_pages=100, max_files=5)