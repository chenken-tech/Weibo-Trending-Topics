#
# import os
# import re
# import json
# import time
# import random
# import pandas as pd
# from pathlib import Path
#
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
#
#
# ROOT_DIR = r"E:/dataset/微博热搜数据集"
#
# USER_ID_COL = "user_id"
# CACHE_JSON = "user_info_cache.json"
#
# HEADLESS = False
#
# FAST_WAIT_TOTAL = 2.5
# FAST_WAIT_INTERVAL = 0.25
#
# MIN_SLEEP = 0.2
# MAX_SLEEP = 0.5
#
# COOKIE_STR = """
# <在此填入你自己的微博 Cookie，或改用 os.environ.get("WEIBO_COOKIE")>
# """.strip()
#
#
# def parse_cookie_string(cookie_str):
#     cookies = []
#
#     for part in cookie_str.split(";"):
#         part = part.strip()
#         if not part or "=" not in part:
#             continue
#
#         name, value = part.split("=", 1)
#
#         cookies.append({
#             "name": name.strip(),
#             "value": value.strip(),
#             "domain": ".weibo.com",
#             "path": "/"
#         })
#
#     return cookies
#
#
# def add_cookies(driver, cookie_str):
#     driver.get("https://weibo.com/")
#     time.sleep(1)
#
#     driver.delete_all_cookies()
#
#     for cookie in parse_cookie_string(cookie_str):
#         try:
#             driver.add_cookie(cookie)
#         except Exception as e:
#             print(f"添加Cookie失败: {cookie.get('name')}，原因: {e}")
#
#     driver.get("https://weibo.com/")
#     time.sleep(2)
#
#     body_text = driver.find_element(By.TAG_NAME, "body").text
#     print("Cookie写入后页面前200字：")
#     print(body_text[:200])
#
#
# def create_driver():
#     options = Options()
#
#     if HEADLESS:
#         options.add_argument("--headless=new")
#
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--start-maximized")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#
#     prefs = {
#         "profile.managed_default_content_settings.images": 2,
#         "profile.default_content_setting_values.notifications": 2,
#     }
#     options.add_experimental_option("prefs", prefs)
#
#     driver = webdriver.Chrome(
#         service=Service(ChromeDriverManager().install()),
#         options=options
#     )
#
#     driver.set_page_load_timeout(20)
#     return driver
#
#
# def normalize_user_id(x):
#     x = str(x).strip()
#
#     if x.endswith(".0"):
#         x = x[:-2]
#
#     if x == "" or x.lower() == "nan":
#         return ""
#
#     return x
#
#
# def build_user_url(user_id):
#     user_id = normalize_user_id(user_id)
#
#     if not user_id:
#         return ""
#
#     return f"https://weibo.com/u/{user_id}"
#
#
# def normalize_count(text):
#     if not text:
#         return ""
#
#     text = str(text).strip()
#     text = text.replace(",", "").replace(" ", "")
#
#     m = re.search(r"(\d+\.?\d*)\s*(万|亿)?", text)
#
#     if not m:
#         return ""
#
#     num = float(m.group(1))
#     unit = m.group(2)
#
#     if unit == "万":
#         num *= 10000
#     elif unit == "亿":
#         num *= 100000000
#
#     return str(int(num))
#
#
# def extract_metrics_from_text(page_text):
#     result = {
#         "主页关注数": "",
#         "主页粉丝数": "",
#         "主页转评赞数": ""
#     }
#
#     follow_match = re.search(r"([\d\.]+(?:万|亿)?)\s*关注", page_text)
#     fan_match = re.search(r"([\d\.]+(?:万|亿)?)\s*粉丝", page_text)
#     interact_match = re.search(r"([\d\.]+(?:万|亿)?)\s*转评赞", page_text)
#
#     if follow_match:
#         result["主页关注数"] = normalize_count(follow_match.group(1))
#
#     if fan_match:
#         result["主页粉丝数"] = normalize_count(fan_match.group(1))
#
#     if interact_match:
#         result["主页转评赞数"] = normalize_count(interact_match.group(1))
#
#     return result
#
#
# def has_metrics(info):
#     return bool(
#         info.get("主页关注数")
#         or info.get("主页粉丝数")
#         or info.get("主页转评赞数")
#     )
#
#
# def fast_get_body_text(driver):
#     start = time.time()
#     last_text = ""
#
#     while time.time() - start < FAST_WAIT_TOTAL:
#         try:
#             text = driver.find_element(By.TAG_NAME, "body").text
#             last_text = text
#
#             if ("关注" in text and "粉丝" in text) or "转评赞" in text:
#                 return text
#
#         except Exception:
#             pass
#
#         time.sleep(FAST_WAIT_INTERVAL)
#
#     return last_text
#
#
# def fetch_user_metrics(driver, user_id):
#     user_id = normalize_user_id(user_id)
#     user_url = build_user_url(user_id)
#
#     result = {
#         "用户主页url": user_url,
#         "主页关注数": "",
#         "主页粉丝数": "",
#         "主页转评赞数": "",
#         "主页抓取状态": "失败"
#     }
#
#     if not user_url:
#         result["主页抓取状态"] = "user_id为空"
#         return result
#
#     try:
#         driver.get(user_url)
#         page_text = fast_get_body_text(driver)
#
#         metrics = extract_metrics_from_text(page_text)
#         result.update(metrics)
#
#         if has_metrics(result):
#             result["主页抓取状态"] = "成功"
#         else:
#             result["主页抓取状态"] = "未匹配到主页数据"
#
#     except Exception as e:
#         result["主页抓取状态"] = f"异常: {str(e)[:80]}"
#
#     return result
#
#
# def read_csv_safely(csv_path):
#     for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
#         try:
#             return pd.read_csv(csv_path, encoding=enc)
#         except Exception:
#             pass
#
#     raise RuntimeError(f"CSV读取失败: {csv_path}")
#
#
# def find_csv_files(root_dir):
#     csv_paths = []
#
#     for topic_name in os.listdir(root_dir):
#         topic_dir = os.path.join(root_dir, topic_name)
#
#         if not os.path.isdir(topic_dir):
#             continue
#
#         for file in os.listdir(topic_dir):
#             if not file.lower().endswith(".csv"):
#                 continue
#
#             if file.lower().endswith("_scored.csv"):
#                 continue
#
#             if file.lower().endswith("_selected.csv"):
#                 continue
#
#             csv_paths.append(os.path.join(topic_dir, file))
#
#     return csv_paths
#
# def load_cache():
#     if not os.path.exists(CACHE_JSON):
#         return {}
#
#     with open(CACHE_JSON, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def save_cache(cache):
#     with open(CACHE_JSON, "w", encoding="utf-8") as f:
#         json.dump(cache, f, ensure_ascii=False, indent=2)
#
#
# def collect_all_user_ids(csv_paths):
#     all_ids = set()
#     csv_dfs = {}
#
#     for csv_path in csv_paths:
#         df = read_csv_safely(csv_path)
#         csv_dfs[csv_path] = df
#
#         if USER_ID_COL not in df.columns:
#             print(f"[跳过收集] 没有 {USER_ID_COL}: {csv_path}")
#             continue
#
#         ids = df[USER_ID_COL].dropna().apply(normalize_user_id)
#         ids = [x for x in ids if x]
#         all_ids.update(ids)
#
#     return sorted(all_ids), csv_dfs
#
#
# def crawl_unique_users(driver, user_ids, cache):
#     total = len(user_ids)
#
#     for i, user_id in enumerate(user_ids, 1):
#         if user_id in cache and cache[user_id].get("主页抓取状态") == "成功":
#             print(f"[缓存] {i}/{total} user_id={user_id}")
#             continue
#
#         print(f"[抓取] {i}/{total} user_id={user_id}")
#
#         info = fetch_user_metrics(driver, user_id)
#         cache[user_id] = info
#
#         print(
#             f"  状态={info.get('主页抓取状态')}, "
#             f"关注={info.get('主页关注数')}, "
#             f"粉丝={info.get('主页粉丝数')}, "
#             f"转评赞={info.get('主页转评赞数')}"
#         )
#
#         if i % 20 == 0:
#             save_cache(cache)
#
#         time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
#
#     save_cache(cache)
#
#
# def write_back_csvs(csv_dfs, cache):
#     """
#     直接把三列用户特征写回原CSV：
#     主页关注数、主页粉丝数、主页转评赞数
#
#     不生成新文件，不添加用户主页url，不添加主页抓取状态。
#     """
#     for csv_path, df in csv_dfs.items():
#         if USER_ID_COL not in df.columns:
#             continue
#
#         follow_list = []
#         fan_list = []
#         interact_list = []
#
#         for _, row in df.iterrows():
#             user_id = normalize_user_id(row[USER_ID_COL])
#
#             info = cache.get(user_id, {})
#
#             follow_list.append(info.get("主页关注数", ""))
#             fan_list.append(info.get("主页粉丝数", ""))
#             interact_list.append(info.get("主页转评赞数", ""))
#
#         # 如果原来已有这三列，直接覆盖；没有则追加到最后
#         df["主页关注数"] = follow_list
#         df["主页粉丝数"] = fan_list
#         df["主页转评赞数"] = interact_list
#
#         # 直接覆盖原CSV
#         df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#
#         print(f"[已写回原文件] {csv_path}")
#
#
# def main():
#     csv_paths = find_csv_files(ROOT_DIR)
#     print(f"发现CSV数量: {len(csv_paths)}")
#
#     user_ids, csv_dfs = collect_all_user_ids(csv_paths)
#     print(f"唯一 user_id 数量: {len(user_ids)}")
#
#     cache = load_cache()
#     print(f"已加载缓存数量: {len(cache)}")
#
#     driver = create_driver()
#
#     try:
#         add_cookies(driver, COOKIE_STR)
#
#         print("Cookie已写入，开始抓取。")
#         crawl_unique_users(driver, user_ids, cache)
#
#         print("开始回填CSV。")
#         write_back_csvs(csv_dfs, cache)
#
#     finally:
#         driver.quit()
#
#     print("\n全部处理完成")
#
#
# if __name__ == "__main__":
#     main()




import os
import re
import json
import time
import random
import tempfile
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# =========================
# 1. 配置区（按需修改）
# =========================
ROOT_DIR = r"D:\py project\weibo-search\结果文件"

USER_ID_COL = "user_id"
CACHE_JSON = "user_info_cache.json"          # 全局缓存文件（最终合并结果）
TEMP_CACHE_PREFIX = "user_info_cache_worker_"  # 临时缓存前缀

HEADLESS = True      # 并行时建议开启无头，节省资源
PARALLEL_WORKERS = 4 # Selenium 并发建议3，避免ChromeDriver竞争

# 抓取速度参数
FAST_WAIT_TOTAL = 2.5
FAST_WAIT_INTERVAL = 0.25
MIN_SLEEP = 0.2
MAX_SLEEP = 0.5

# 微博 Cookie（请替换成自己的有效 Cookie）
# 微博 Cookie：填入你自己的整串 Cookie，或设置环境变量 WEIBO_COOKIE。
COOKIE_STR = os.environ.get("WEIBO_COOKIE", "").strip()


# =========================
# 2. 浏览器与 Cookie 辅助函数
# =========================
def parse_cookie_string(cookie_str):
    cookies = []
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookies.append({
            "name": name.strip(),
            "value": value.strip(),
            "domain": ".weibo.com",
            "path": "/"
        })
    return cookies


def add_cookies(driver, cookie_str):
    driver.get("https://weibo.com/")
    time.sleep(1)
    driver.delete_all_cookies()
    for cookie in parse_cookie_string(cookie_str):
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"添加Cookie失败: {cookie.get('name')}，原因: {e}")
    driver.get("https://weibo.com/")
    time.sleep(2)
    # 可选：打印页面前200字符验证登录状态
    # body_text = driver.find_element(By.TAG_NAME, "body").text
    # print("Cookie写入后页面前200字：", body_text[:200])


def create_driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 禁用图片、通知等加速
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.set_page_load_timeout(20)
    return driver


# =========================
# 3. 数据提取相关函数
# =========================
def normalize_user_id(x):
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    if x == "" or x.lower() == "nan":
        return ""
    return x


def build_user_url(user_id):
    user_id = normalize_user_id(user_id)
    if not user_id:
        return ""
    return f"https://weibo.com/u/{user_id}"


def normalize_count(text):
    if not text:
        return ""
    text = str(text).strip().replace(",", "").replace(" ", "")
    m = re.search(r"(\d+\.?\d*)\s*(万|亿)?", text)
    if not m:
        return ""
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "万":
        num *= 10000
    elif unit == "亿":
        num *= 100000000
    return str(int(num))


def extract_metrics_from_text(page_text):
    result = {"主页关注数": "", "主页粉丝数": "", "主页转评赞数": ""}
    follow_match = re.search(r"([\d\.]+(?:万|亿)?)\s*关注", page_text)
    fan_match = re.search(r"([\d\.]+(?:万|亿)?)\s*粉丝", page_text)
    interact_match = re.search(r"([\d\.]+(?:万|亿)?)\s*转评赞", page_text)
    if follow_match:
        result["主页关注数"] = normalize_count(follow_match.group(1))
    if fan_match:
        result["主页粉丝数"] = normalize_count(fan_match.group(1))
    if interact_match:
        result["主页转评赞数"] = normalize_count(interact_match.group(1))
    return result


def has_metrics(info):
    return bool(info.get("主页关注数") or info.get("主页粉丝数") or info.get("主页转评赞数"))


def fast_get_body_text(driver):
    start = time.time()
    last_text = ""
    while time.time() - start < FAST_WAIT_TOTAL:
        try:
            text = driver.find_element(By.TAG_NAME, "body").text
            last_text = text
            if ("关注" in text and "粉丝" in text) or "转评赞" in text:
                return text
        except Exception:
            pass
        time.sleep(FAST_WAIT_INTERVAL)
    return last_text


def fetch_user_metrics(driver, user_id):
    user_id = normalize_user_id(user_id)
    user_url = build_user_url(user_id)
    result = {
        "用户主页url": user_url,
        "主页关注数": "",
        "主页粉丝数": "",
        "主页转评赞数": "",
        "主页抓取状态": "失败"
    }
    if not user_url:
        result["主页抓取状态"] = "user_id为空"
        return result
    try:
        driver.get(user_url)
        page_text = fast_get_body_text(driver)
        metrics = extract_metrics_from_text(page_text)
        result.update(metrics)
        result["主页抓取状态"] = "成功" if has_metrics(result) else "未匹配到主页数据"
    except Exception as e:
        result["主页抓取状态"] = f"异常: {str(e)[:80]}"
    return result


# =========================
# 4. CSV 读取与文件查找
# =========================
def read_csv_safely(csv_path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV读取失败: {csv_path}")


def find_csv_files(root_dir):
    csv_paths = []
    for topic_name in os.listdir(root_dir):
        topic_dir = os.path.join(root_dir, topic_name)
        if not os.path.isdir(topic_dir):
            continue
        for file in os.listdir(topic_dir):
            if not file.lower().endswith(".csv"):
                continue
            if file.lower().endswith("_scored.csv") or file.lower().endswith("_selected.csv"):
                continue
            csv_paths.append(os.path.join(topic_dir, file))
    return csv_paths


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache, cache_path):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# =========================
# 5. Worker 进程：抓取分配给它的 user_id 子集
# =========================
def crawl_user_ids_worker(user_ids_subset, input_cache_path, output_cache_path, worker_id):
    """
    子进程入口：加载已有缓存，创建 driver，抓取所有分配的 user_id，
    将结果（包含旧缓存 + 新抓取）写入 output_cache_path。
    """
    print(f"[Worker {worker_id}] 启动，分配 {len(user_ids_subset)} 个 user_id")
    cache = load_cache(input_cache_path)        # 加载全局缓存（只读）
    driver = create_driver()
    try:
        add_cookies(driver, COOKIE_STR)
        # 不需要再次确认登录，无头模式下直接开始
        total = len(user_ids_subset)
        for idx, user_id in enumerate(user_ids_subset, 1):
            if user_id in cache and cache[user_id].get("主页抓取状态") == "成功":
                print(f"[Worker {worker_id}] 缓存命中 {idx}/{total} user_id={user_id}")
                continue
            print(f"[Worker {worker_id}] 抓取 {idx}/{total} user_id={user_id}")
            info = fetch_user_metrics(driver, user_id)
            cache[user_id] = info
            print(f"  状态={info['主页抓取状态']}, 关注={info['主页关注数']}, 粉丝={info['主页粉丝数']}, 转评赞={info['主页转评赞数']}")
            if idx % 20 == 0:
                save_cache(cache, output_cache_path)   # 中途保存
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
        # 最终保存
        save_cache(cache, output_cache_path)
    finally:
        driver.quit()
    print(f"[Worker {worker_id}] 完成，缓存已写入 {output_cache_path}")


# =========================
# 6. 合并所有 worker 的临时缓存
# =========================
def merge_worker_caches(worker_output_paths, final_cache_path):
    """
    合并 worker 缓存。
    增加空文件和 JSON 损坏保护，避免单个 worker 异常导致整体失败。
    """
    merged = {}

    for path in worker_output_paths:
        if not os.path.exists(path):
            print(f"[跳过] 缓存文件不存在: {path}")
            continue

        if os.path.getsize(path) == 0:
            print(f"[跳过] 空缓存文件: {path}")
            continue

        try:
            part = load_cache(path)
            if isinstance(part, dict):
                merged.update(part)
                print(f"[合并成功] {path}, 数量={len(part)}")
            else:
                print(f"[跳过] 非字典缓存: {path}")

        except Exception as e:
            print(f"[缓存读取失败] {path}")
            print(f"原因: {e}")
            continue

    save_cache(merged, final_cache_path)
    return merged


# =========================
# 7. 回填 CSV（单线程，避免文件冲突）
# =========================
def write_back_csvs(csv_dfs, cache):
    """
    将三列用户特征直接写回原 CSV 文件（覆盖原文件）。
    """
    for csv_path, df in csv_dfs.items():
        if USER_ID_COL not in df.columns:
            continue
        follow_list, fan_list, interact_list = [], [], []
        for _, row in df.iterrows():
            user_id = normalize_user_id(row[USER_ID_COL])
            info = cache.get(user_id, {})
            follow_list.append(info.get("主页关注数", ""))
            fan_list.append(info.get("主页粉丝数", ""))
            interact_list.append(info.get("主页转评赞数", ""))
        df["主页关注数"] = follow_list
        df["主页粉丝数"] = fan_list
        df["主页转评赞数"] = interact_list
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[写回] {csv_path}")


# =========================
# 8. 主函数（协调并行）
# =========================
def main():
    # 1. 收集所有 CSV 文件及所有 user_id
    csv_paths = find_csv_files(ROOT_DIR)
    print(f"发现 CSV 数量: {len(csv_paths)}")
    user_ids, csv_dfs = collect_all_user_ids(csv_paths)
    print(f"唯一 user_id 数量: {len(user_ids)}")
    if not user_ids:
        print("没有找到任何 user_id，退出。")
        return

    # 2. 加载已有全局缓存（如果存在）
    global_cache_path = CACHE_JSON
    try:
        existing_cache = load_cache(global_cache_path)
    except Exception as e:
        print(f"[警告] 全局缓存读取失败，将重新建立缓存: {e}")
        existing_cache = {}

    print(f"已加载全局缓存数量: {len(existing_cache)}")
    # 为了不让每个 worker 重复抓取已成功的 id，将现有缓存保存为一个临时文件供所有 worker 读取
    # 注意：若 global_cache_path 已经存在，直接用它作为 input_cache_path；但 worker 会只读它。
    # 为了避免多进程同时写同一个文件，我们让每个 worker 写自己的临时文件，最后合并。
    input_cache_for_workers = global_cache_path  # 每个 worker 会只读这个文件

    # 3. 决定并行进程数
    num_workers = min(PARALLEL_WORKERS, len(user_ids), mp.cpu_count() or 4)
    print(f"使用 {num_workers} 个并行进程")

    # 4. 分割 user_id 列表
    chunk_size = (len(user_ids) + num_workers - 1) // num_workers
    user_id_chunks = [user_ids[i:i+chunk_size] for i in range(0, len(user_ids), chunk_size)]

    # 5. 为每个 worker 准备临时输出缓存文件
    temp_files = []
    for i in range(len(user_id_chunks)):
        fd, temp_path = tempfile.mkstemp(prefix=TEMP_CACHE_PREFIX, suffix=".json", text=True)
        os.close(fd)
        temp_files.append(temp_path)

    # 6. 启动进程池
    processes = []
    for i, (chunk, out_path) in enumerate(zip(user_id_chunks, temp_files)):
        p = mp.Process(target=crawl_user_ids_worker, args=(chunk, input_cache_for_workers, out_path, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        print(
            f"[Worker结束] PID={p.pid}, exitcode={p.exitcode}"
        )

    # 7. 合并所有 worker 的临时缓存，生成最终全量缓存
    print("合并各 worker 缓存...")
    final_cache = merge_worker_caches(temp_files, global_cache_path)
    print(f"合并后总缓存数量: {len(final_cache)}")

    # 8. 回填 CSV（单线程）
    print("开始回填 CSV 文件...")
    write_back_csvs(csv_dfs, final_cache)

    # 9. 清理临时文件
    for tmp in temp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print("\n全部处理完成！")


def collect_all_user_ids(csv_paths):
    """收集所有 CSV 中的 user_id（去重）并同时缓存 DataFrame 对象"""
    all_ids = set()
    csv_dfs = {}
    for csv_path in csv_paths:
        df = read_csv_safely(csv_path)
        csv_dfs[csv_path] = df
        if USER_ID_COL not in df.columns:
            print(f"[跳过收集] 没有 {USER_ID_COL}: {csv_path}")
            continue
        ids = df[USER_ID_COL].dropna().apply(normalize_user_id)
        ids = [x for x in ids if x]
        all_ids.update(ids)
    return sorted(all_ids), csv_dfs


if __name__ == "__main__":
    # 多进程在 Windows 上需要此保护
    mp.freeze_support()
    main()
