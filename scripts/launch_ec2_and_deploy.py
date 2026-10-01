"""AWS EC2 GPU Instance Launcher & Automated Deployment Tool.

Automatically launches a g5.xlarge (NVIDIA A10G 24GB VRAM) instance with 200GB gp3 storage,
waits for instance readiness, associates the transcriprtion key pair, and executes SSH deployment.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import boto3

_ROOT = Path(__file__).resolve().parents[1]
PEM_PATH = _ROOT / "transcriprtion.pem"


def launch_g5_gpu_instance(
    region_name: str = "us-east-1",
    instance_type: str = "g5.xlarge",
    disk_gb: int = 200,
    key_name: str = "transcriprtion",
) -> dict[str, str]:
    """Launch a GPU EC2 instance with 200GB gp3 EBS volume."""
    print(f"Connecting to AWS EC2 in region: {region_name}...")
    ec2_client = boto3.client("ec2", region_name=region_name)
    ec2_resource = boto3.resource("ec2", region_name=region_name)

    # 1. Locate Deep Learning AMI (Ubuntu 22.04 or 20.04 with PyTorch/CUDA)
    print("Finding official AWS Deep Learning AMI...")
    images = ec2_client.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["*Deep Learning Base AMI (Ubuntu 22.04)*", "*Deep Learning AMI GPU PyTorch*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )["Images"]

    if images:
        images.sort(key=lambda x: x["CreationDate"], reverse=True)
        ami_id = images[0]["ImageId"]
        ami_name = images[0]["Name"]
        print(f"Found AMI: {ami_id} ({ami_name})")
    else:
        # Fallback to standard Ubuntu 22.04 LTS AMI
        ubuntu_images = ec2_client.describe_images(
            Owners=["099720109477"],
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]},
            ],
        )["Images"]
        ubuntu_images.sort(key=lambda x: x["CreationDate"], reverse=True)
        ami_id = ubuntu_images[0]["ImageId"]
        ami_name = ubuntu_images[0]["Name"]
        print(f"Using Ubuntu 22.04 LTS AMI: {ami_id} ({ami_name})")

    # 2. Get Default VPC & Subnet
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        vpcs = ec2_client.describe_vpcs()["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]

    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    subnet_id = subnets[0]["SubnetId"]

    # 3. Security Group for SSH
    sg_name = "stt-gpu-sg"
    try:
        sgs = ec2_client.describe_security_groups(Filters=[{"Name": "group-name", "Values": [sg_name]}])["SecurityGroups"]
        if sgs:
            sg_id = sgs[0]["GroupId"]
        else:
            sg = ec2_client.create_security_group(
                GroupName=sg_name,
                Description="Security group for Indian STT GPU EC2 instance",
                VpcId=vpc_id,
            )
            sg_id = sg["GroupId"]
            ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
                    }
                ],
            )
    except Exception as e:
        print(f"Warning setting up security group: {e}")
        sg_id = None

    # 4. Launch EC2 GPU Instance
    print(f"Launching EC2 instance type '{instance_type}' with {disk_gb}GB gp3 disk...")
    block_device_mappings = [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": disk_gb,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }
    ]

    kwargs = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": block_device_mappings,
    }
    if sg_id:
        kwargs["SecurityGroupIds"] = [sg_id]

    instance = ec2_resource.create_instances(**kwargs)[0]
    print(f"Instance requested! ID: {instance.id}. Waiting for running state...")

    instance.wait_until_running()
    instance.reload()

    public_ip = instance.public_ip_address
    public_dns = instance.public_dns_name

    info = {
        "instance_id": instance.id,
        "public_ip": public_ip,
        "public_dns": public_dns,
        "private_ip": instance.private_ip_address,
        "instance_type": instance.instance_type,
        "availability_zone": instance.placement["AvailabilityZone"],
        "region": region_name,
    }

    print("\n✓ AWS EC2 Instance is RUNNING!")
    print(f"  Instance ID:       {info['instance_id']}")
    print(f"  Public IP:         {info['public_ip']}")
    print(f"  Public DNS:        {info['public_dns']}")
    print(f"  Instance Type:     {info['instance_type']}")
    print(f"  Availability Zone: {info['availability_zone']}")

    return info


if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else "us-east-1"
    try:
        launch_g5_gpu_instance(region_name=region)
    except Exception as e:
        print(f"Error launching instance: {e}")
        sys.exit(1)
