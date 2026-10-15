import json
import boto3
import os

INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID", "i-095d56c1504424db7")
REGION = os.environ.get("AWS_REGION", "ap-northeast-1")

ec2 = boto3.client("ec2", region_name=REGION)

def lambda_handler(event, context):
    """
    AWS Lambda 中継ハンドラー:
    先方PCに認証キーを持たせず、EC2 GPUサーバーの安全な起動・停止・IP取得を行います。
    """
    path = event.get("rawPath", "") or event.get("path", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "") or event.get("httpMethod", "")

    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
    }

    if method == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": "{}"}

    try:
        # インスタンス状態取得
        res = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        inst = res["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress", None)

        if "/start" in path:
            if state != "running":
                ec2.start_instances(InstanceIds=[INSTANCE_ID])
                state = "pending"
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"status": "starting", "instance_id": INSTANCE_ID, "state": state, "ip": ip})
            }

        elif "/stop" in path:
            if state == "running":
                ec2.stop_instances(InstanceIds=[INSTANCE_ID])
                state = "stopping"
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"status": "stopping", "instance_id": INSTANCE_ID, "state": state})
            }

        else: # /status
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"instance_id": INSTANCE_ID, "state": state, "ip": ip})
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": str(e)})
        }
