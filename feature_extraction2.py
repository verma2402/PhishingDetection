import re
import socket
import ipaddress
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime, date
from dateutil.parser import parse as date_parse
from googlesearch import search


def diff_month(d1: date, d2: date) -> int:
    """Calculate number of months between two dates."""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def generate_data_set(url: str) -> list:
    """
    Generate dataset by extracting 30 phishing detection features from the URL.
    Returns a list of feature values (1, 0, -1).
    """
    data_set = []

    # 0. Normalize URL
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    # 1. HTTP response and Soup
    try:
        resp = requests.get(url, timeout=10)
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        resp = None
        soup = None

    # 2. Extract domain
    try:
        domain = re.findall(r"://([^/]+)/?", url)[0].lower()
        if domain.startswith('www.'):
            domain = domain[4:]
    except Exception:
        domain = ''

    # 3. WHOIS lookup
    try:
        whois_resp = whois.whois(domain)
    except Exception:
        whois_resp = None

    # 4. Global rank via external service
    try:
        rank_resp = requests.post(
            "https://www.checkpagerank.net/index.php", timeout=10,
            data={"name": domain}
        )
        global_rank = int(re.search(r"Global Rank: ([0-9]+)", rank_resp.text).group(1))
    except Exception:
        global_rank = -1

    # Feature 1: Having IP Address in URL
    try:
        _ = ipaddress.ip_address(domain)
        data_set.append(-1)
    except Exception:
        data_set.append(1)

    # Feature 2: URL Length
    length = len(url)
    if length < 54:
        data_set.append(1)
    elif 54 <= length <= 75:
        data_set.append(0)
    else:
        data_set.append(-1)

    # Feature 3: Shortening Service
    short_pattern = re.compile(
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs",
        re.IGNORECASE
    )
    data_set.append(-1 if short_pattern.search(url) else 1)

    # Feature 4: '@' Symbol
    data_set.append(-1 if '@' in url else 1)

    # Feature 5: '//' Redirect
    slash_positions = [m.start() for m in re.finditer('//', url)]
    if slash_positions and slash_positions[-1] > 6:
        data_set.append(-1)
    else:
        data_set.append(1)

    # Feature 6: Prefix/Suffix '-'
    data_set.append(-1 if re.search(r"https?://[\w\-]+-[\w\-]+/", url) else 1)

    # Feature 7: Subdomain count
    dot_count = url.count('.')
    if dot_count == 1:
        data_set.append(1)
    elif dot_count == 2:
        data_set.append(0)
    else:
        data_set.append(-1)

    # Feature 8: SSL final state (HTTPS)
    data_set.append(1 if url.lower().startswith('https://') else -1)

    # Feature 9: Domain registration length
    try:
        exp = whois_resp.expiration_date
        exp_date = exp[0] if isinstance(exp, list) else exp
        days = (exp_date - date.today()).days
        data_set.append(-1 if days <= 365 else 1)
    except Exception:
        data_set.append(-1)

    # Feature 10: Favicon domain
    if soup:
        try:
            icon = soup.find('link', rel=lambda x: x and 'icon' in x)
            href = icon['href'] if icon else ''
            data_set.append(1 if domain in href or href.startswith('/') else -1)
        except Exception:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 11: Non-standard port
    data_set.append(-1 if ':' in domain and domain.split(':')[1].isdigit() else 1)

    # Feature 12: HTTPS token in URL
    data_set.append(-1 if 'https://' in url.lower().replace('://', '') else 1)

    # Feature 13: Request URL in tags
    if soup:
        total, success = 0, 0
        for tag in soup.find_all(['img', 'audio', 'embed', 'iframe'], src=True):
            total += 1
            src = tag['src']
            if domain in src or src.startswith('/') or src.count('.') == 1:
                success += 1
        percent = (success / total * 100) if total else 0
        if percent < 22:
            data_set.append(1)
        elif percent < 61:
            data_set.append(0)
        else:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 14: URL of Anchor tags
    if soup:
        total, unsafe = 0, 0
        for a in soup.find_all('a', href=True):
            total += 1
            href = a['href']
            if any(x in href.lower() for x in ['#', 'javascript', 'mailto']) or not (domain in href or href.startswith('/')):
                unsafe += 1
        percent = (unsafe / total * 100) if total else 0
        if percent < 31:
            data_set.append(1)
        elif percent < 67:
            data_set.append(0)
        else:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 15: Links in <link> and <script>
    if soup:
        total, success = 0, 0
        for tag in soup.find_all(['link', 'script'], href=True):
            total += 1
            url_ref = tag.get('href') or tag.get('src', '')
            if domain in url_ref or url_ref.startswith('/') or url_ref.count('.') == 1:
                success += 1
        percent = (success / total * 100) if total else 0
        if percent < 17:
            data_set.append(1)
        elif percent < 81:
            data_set.append(0)
        else:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 16: Server Form Handler (SFH)
    if soup:
        try:
            forms = soup.find_all('form', action=True)
            action = forms[0]['action'] if forms else ''
            if not action or action == 'about:blank':
                data_set.append(-1)
            elif domain not in action and not action.startswith('/'):
                data_set.append(0)
            else:
                data_set.append(1)
        except Exception:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 17: Submitting to email
    if resp and resp.text:
        data_set.append(-1 if re.search(r"mailto:|mail\(" , resp.text, re.IGNORECASE) else 1)
    else:
        data_set.append(-1)

    # Feature 18: Abnormal URL (empty page)
    data_set.append(-1 if resp and resp.text.strip() else 1)

    # Feature 19: Redirect count
    if resp:
        hops = len(resp.history)
        if hops <= 1:
            data_set.append(-1)
        elif hops <= 4:
            data_set.append(0)
        else:
            data_set.append(1)
    else:
        data_set.append(-1)

    # Feature 20: onmouseover script
    data_set.append(1 if resp and re.search(r"onmouseover=", resp.text) else -1)

    # Feature 21: Right click disabled
    data_set.append(1 if resp and re.search(r"event\.button ?== ?2", resp.text) else -1)

    # Feature 22: popup window alert
    data_set.append(1 if resp and re.search(r"alert\(", resp.text) else -1)

    # Feature 23: IFrame detection
    data_set.append(1 if resp and re.search(r"<iframe|<frameBorder", resp.text) else -1)

    # Feature 24: Age of domain
    if whois_resp:
        try:
            reg_date = whois_resp.creation_date
            reg = reg_date[0] if isinstance(reg_date, list) else reg_date
            months = diff_month(date.today(), date_parse(str(reg)))
            data_set.append(1 if months >= 6 else -1)
        except Exception:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 25: DNS record validity
    try:
        d = whois_resp
        days = (d.expiration_date - d.creation_date).days
        data_set.append(-1 if days <= 365 else 1)
    except Exception:
        data_set.append(-1)

    # Feature 26: Web traffic via Alexa
    try:
        xml = requests.get(f"http://data.alexa.com/data?cli=10&dat=s&url={url}", timeout=10).content
        rank = int(BeautifulSoup(xml, 'xml').find('REACH')['RANK'])
        data_set.append(1 if rank < 100000 else 0)
    except Exception:
        data_set.append(-1)

    # Feature 27: Page rank placeholder
    data_set.append(1 if global_rank < 0 or global_rank > 100000 else -1)

    # Feature 28: Google index
    try:
        sites = search(url, num_results=5)
        data_set.append(1 if sites else -1)
    except Exception:
        data_set.append(-1)

    # Feature 29: Links pointing to page
    if resp and resp.text:
        links = re.findall(r"<a href=", resp.text)
        count = len(links)
        if count == 0:
            data_set.append(1)
        elif count <= 2:
            data_set.append(0)
        else:
            data_set.append(-1)
    else:
        data_set.append(-1)

    # Feature 30: Statistical report (IP/URL match)
    try:
        ip_addr = socket.gethostbyname(domain)
        url_match = re.search(r"at\.ua|usa\.cc|sweddy\.com", url)
        ip_match = re.search(r"146\.112\.61\.108|213\.174\.157\.151", ip_addr)
        data_set.append(-1 if url_match or ip_match else 1)
    except Exception:
        data_set.append(-1)

    return data_set
