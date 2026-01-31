
import os
import re
import requests
import tarfile
import json

API_URL = "https://data.aiops.cn/api/fs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
    "Origin": "https://data.aiops.cn",  # 替换为实际来源页面
    "Content-Type": "application/json;charset=UTF-8",
    "Accept": "application/json, text/plain, */*",
}


def main(api_path, base_dir="./data"):
    """
    主函数，启动爬虫
    :param start_url: 起始URL
    :param base_dir: 根下载目录
    """
    current_dir = os.path.join(base_dir, api_path.replace("-", "")) 
    os.makedirs(current_dir, exist_ok=True)  # 确保根目录存在
    print(f"开始处理: {current_dir}")
    process_api_list(api_path, current_dir)  # 处理API路径


def fetch_api_get(path, password=""):
    payload = {"path": path, "password": password}
    try:
        response = requests.post(
            f"{API_URL}/get", headers=HEADERS, data=json.dumps(payload), timeout=200
        )
        response.raise_for_status()
        return response.json().get("data", {}).get("raw_url", "")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except json.JSONDecodeError:
        print("JSON解析失败")
        return None


def fetch_api_list(path, page=1, per_page=100, password=""):
    """
    从API获取数据
    post:/list
    """
    payload = {
        "path": path,
        "password": password,
        "page": page,
        "per_page": per_page,
        "refresh": False,
    }

    try:
        response = requests.post(
            f"{API_URL}/list", headers=HEADERS, data=json.dumps(payload), timeout=200
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except json.JSONDecodeError:
        print("JSON解析失败")
        return None


def process_api_list(path, current_dir):
    """
    处理API返回的列表数据
    """
    data = fetch_api_list(path)
    if not data or "data" not in data:
        print(f"未找到数据或数据格式错误: {path}")
        return

    items = data["data"].get("content", [])
    print(f"处理路径: {path} - 发现 {len(items)} 个项目")
    for item in items:
        sub_path=f"/{path}/{item['name']}"
        if not item.get("is_dir"):
            file_url = fetch_api_get(sub_path)

            if file_url is not None:
                print(f"下载文件: {sub_path} - {file_url}")
                download_file(file_url, current_dir)
                continue

        new_dir = os.path.join(current_dir, item["name"])
        os.makedirs(new_dir, exist_ok=True)  # 确保目录存在
        print(f"进入子目录: {new_dir}")
        process_api_list(sub_path, new_dir)  # 递归处理子目录


def download_file(file_url, save_dir):
    """下载文件到指定目录"""
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # 从URL提取文件名
        filename = os.path.basename(file_url)
        filename = re.sub(r'\?.*$', '', filename)  # 去除查询参数

        if not filename:
            filename = f"file_{hash(file_url)}.dat"  # 备用命名

        save_path = os.path.join(save_dir, filename)

        # 避免重复下载
        if os.path.exists(save_path):
            print(f"文件已存在: {save_path}")
            tar_file(save_dir, filename)  # 解压缩文件
            return

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"下载完成: {save_path}")

    except Exception as e:
        print(f"下载失败: {file_url} - {str(e)}")

    # if filename.startswith("metric"):
    tar_file(save_dir, filename)  # 解压缩文件


def tar_file(save_dir, file_name):
    """
    解压缩tar.gz文件
    :param file_path: 文件路径
    """
    save_path = os.path.join(save_dir, file_name)

    if not file_name.endswith(".tar.gz"):
        # print(f"文件不是tar.gz格式: {file_name}")
        return

    if not os.path.exists(save_path):
        print(f"文件不存在: {save_path}")
        return

    try:
        with tarfile.open(save_path, "r:gz") as tar:
            tar.extractall(path=save_dir)
            print(f"解压完成: {save_path}")
    except Exception as e:
        print(f"解压失败: {save_path} - {str(e)}")


if __name__ == "__main__":
    # 示例使用
    # for i in range(9, 10):
    #     main(
    #         f"2025-06-0{i}",
    #     )

    for i in range(10, 30):
        # print(i)
        main(
            f"2025-06-{i}",
        )
