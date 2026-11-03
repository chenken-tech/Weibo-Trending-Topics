
import os
import time
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# =========================
# 1. 基本配置
# =========================
ROOT_DIR = r"D:\py project\weibo-search\结果文件"
OUTPUT_FILENAME = "popularity.txt"

BASE_URL = "https://m.s.weibo.com/vtopic/detail_new?click_from=searchpc&q="

# 这里直接填你从浏览器复制出来的整串 Cookie
# 微博 Cookie：请填入你自己的整串 Cookie（从浏览器开发者工具复制），
# 或设置环境变量 WEIBO_COOKIE 后留空此处，避免把私人凭据提交到仓库。
COOKIE_STR = os.environ.get("WEIBO_COOKIE", "").strip()

# 是否无头运行
HEADLESS = False

# 页面等待时间
WAIT_SECONDS = 20


# =========================
# 2. 整串 Cookie 转 Selenium Cookie
# =========================
def parse_cookie_string(cookie_str: str):
    """
    把这种格式：
    SUB=xxx; SUBP=xxx; SSOLoginState=xxx
    转成 Selenium add_cookie 需要的格式
    """
    cookies = []
    if not cookie_str:
        return cookies

    parts = cookie_str.split(";")
    for part in parts:
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            cookies.append({
                "name": name,
                "value": value
            })
    return cookies


# =========================
# 3. 根据文件夹名构造链接
# =========================
def build_topic_url(folder_name: str) -> str:
    folder_name = folder_name.strip()
    if folder_name.startswith("%23") and folder_name.endswith("%23"):
        q = folder_name
    else:
        q = quote(folder_name, safe="")
    return BASE_URL + q


# =========================
# 4. 启动浏览器
# =========================
def create_driver():
    chrome_options = Options()
    if HEADLESS:
        chrome_options.add_argument("--headless=new")

    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    driver.set_page_load_timeout(60)
    return driver


# =========================
# 5. 写入 Cookie
# =========================
def add_cookies(driver, cookie_str: str):
    """
    必须先打开一次微博域名，才能 add_cookie
    """
    cookies = parse_cookie_string(cookie_str)

    driver.get("https://m.s.weibo.com/")
    time.sleep(2)

    for cookie in cookies:
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"添加 cookie 失败: {cookie.get('name')}, 原因: {e}")

    driver.refresh()
    time.sleep(2)

def normalize_count(text: str) -> str:
    """
    把微博计数统一转成纯数字字符串

    例如：
    2175.9万 -> 21759000
    9万 -> 90000
    1.9万 -> 19000
    3088 -> 3088
    """
    if not text:
        return ""

    text = text.strip().replace(",", "").replace(" ", "")

    try:
        if text.endswith("万"):
            num = float(text[:-1])
            return str(int(num * 10000))
        elif text.endswith("亿"):
            num = float(text[:-1])
            return str(int(num * 100000000))
        else:
            # 普通整数或小数
            return str(int(float(text)))
    except Exception:
        # 如果转换失败，就原样返回
        return text
# =========================
# 6. 抓取四项数据
# =========================
def fetch_topic_metrics(driver, url: str) -> dict:
    driver.get(url)

    wait = WebDriverWait(driver, WAIT_SECONDS)
    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.detail-data"))
    )

    result = {
        "阅读量": "",
        "讨论量": "",
        "互动量": "",
        "原创量": ""
    }

    item_cols = driver.find_elements(By.CSS_SELECTOR, "div.detail-data div.item-col")

    for item in item_cols:
        try:
            num_text = item.find_element(By.CSS_SELECTOR, "div.num").text.strip()
            des_text = item.find_element(By.CSS_SELECTOR, "div.des").text.strip()

            if des_text in result:
                # result[des_text] = num_text
                result[des_text] = normalize_count(num_text)
        except Exception:
            continue

    return result


# =========================
# 7. 判断是否抓取到有效数据
# =========================
def has_valid_metrics(metrics: dict) -> bool:
    """
    四项里只要有一项抓到了非空值，就认为有效
    """
    return any(v and str(v).strip() for v in metrics.values())


# =========================
# 8. 保存 txt
# =========================
def save_metrics_to_txt(folder_path: str, metrics: dict):
    output_path = os.path.join(folder_path, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"阅读量：{metrics.get('阅读量', '')}\n")
        f.write(f"讨论量：{metrics.get('讨论量', '')}\n")
        f.write(f"互动量：{metrics.get('互动量', '')}\n")
        f.write(f"原创量：{metrics.get('原创量', '')}\n")


# =========================
# 9. 主流程
# =========================
def main():
    if not os.path.exists(ROOT_DIR):
        print(f"目录不存在：{ROOT_DIR}")
        return

    folder_names = [
        name for name in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, name))
    ]

    print(f"共发现 {len(folder_names)} 个话题文件夹")

    driver = create_driver()

    try:
        add_cookies(driver, COOKIE_STR)

        for idx, folder_name in enumerate(folder_names, 1):
            folder_path = os.path.join(ROOT_DIR, folder_name)
            url = build_topic_url(folder_name)

            print(f"\n[{idx}/{len(folder_names)}] 正在处理：{folder_name}")
            print(f"链接：{url}")

            try:
                metrics = fetch_topic_metrics(driver, url)

                if has_valid_metrics(metrics):
                    save_metrics_to_txt(folder_path, metrics)
                    print("抓取成功：", metrics)
                else:
                    print("未抓取到有效数据，不生成txt文件")

            except Exception as e:
                print(f"处理失败：{e}")
                print("本次不生成txt文件")

            time.sleep(2)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
