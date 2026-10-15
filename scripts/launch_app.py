import os
import sys
import time
import subprocess
import urllib.request
import json

INSTANCE_ID = "i-095d56c1504424db7"
REGION = "ap-northeast-1"
DEFAULT_IP = "54.249.95.222"

def print_banner(msg):
    print("=" * 55)
    print(f"  {msg}")
    print("=" * 55)

def check_aws_cli():
    try:
        res = subprocess.run(["aws", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0
    except FileNotFoundError:
        return False

def get_instance_info(instance_id, region):
    cmd = ["aws", "ec2", "describe-instances", "--instance-ids", instance_id, "--region", region, "--output", "json"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return None, None
    data = json.loads(res.stdout)
    try:
        inst = data["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress", None)
        return state, ip
    except Exception:
        return None, None

def start_ec2_instance(instance_id, region):
    print(f"[AWS] サーバー (ID: {instance_id}) の状態を確認しています...")
    state, ip = get_instance_info(instance_id, region)

    if state == "running" and ip:
        print(f"[AWS] サーバーは既に起動しています (IP: {ip})")
        return ip

    if state != "running":
        print("[AWS] サーバーの電源を自動投入しています (Start Instance)...")
        subprocess.run(["aws", "ec2", "start-instances", "--instance-ids", instance_id, "--region", region], check=True)
        
        print("[AWS] サーバーの起動完了を待機しています (約20〜30秒)...")
        subprocess.run(["aws", "ec2", "wait", "instance-running", "--instance-ids", instance_id, "--region", region], check=True)

    # 起動後の最新IP取得
    for _ in range(10):
        _, ip = get_instance_info(instance_id, region)
        if ip:
            break
        time.sleep(2)

    print(f"[AWS] サーバーが起動しました！(新IP: {ip})")
    return ip

def wait_for_server_ready(ip, port=8080, timeout=60):
    url = f"http://{ip}:{port}/api/status"
    print(f"[接続確認] AIサービスが準備完了するのを待機しています ({url})...")
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=2)
            if req.status == 200:
                print("[接続確認] サーバー準備完了！アプリ画面を起動します。")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def launch_chrome_app(url):
    print(f"[アプリ起動] {url}")
    cmd = [
        "open", "-na", "Google Chrome", "--args",
        f"--app={url}",
        "--window-size=1360,860",
        "--user-data-dir=/tmp/livechat_app_profile",
        f"--unsafely-treat-insecure-origin-as-secure={url}"
    ]
    subprocess.Popen(cmd)

def main():
    print_banner("LiveChat AI Studio - 自動サーバー連動ランチャー")

    has_aws = check_aws_cli()
    server_ip = DEFAULT_IP

    if has_aws:
        try:
            ip = start_ec2_instance(INSTANCE_ID, REGION)
            if ip:
                server_ip = ip
        except Exception as e:
            print(f"[AWS起動警告] 自動起動スクリプト失敗: {e} (固定IPで接続を試みます)")
    else:
        print("[案内] AWS CLIが未設定のため、直前IPで直接接続します。")

    server_url = f"http://{server_ip}:8080"
    
    # サーバーのポート待機 (最大30秒)
    wait_for_server_ready(server_ip, 8080, timeout=40)

    # アプリ起動
    launch_chrome_app(server_url)

if __name__ == "__main__":
    main()
