import boto3
import json

iam = boto3.client('iam')
s3 = boto3.client('s3')

bucket_name = "sagemaker-us-east-1-436090206346"

# Step 1: Create an Iam role
role_name = "SageMakerExecutionRole"
assume_role_policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

role_response = iam.create_role(
    RoleName=role_name,
    AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
)

# Step 2: Create a policy granting access to the S3 bucket
s3_access_policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{}/*".format(bucket_name),
                "arn:aws:s3:::{}".format(bucket_name)
            ]
        }
    ]
}

policy_response = iam.create_policy(
    PolicyName="S3AccessPolicy",
    PolicyDocument=json.dumps(s3_access_policy_document)
)

# Step 3: Attach the policy to the role
iam.attach_role_policy(
    RoleName=role_name,
    PolicyArn=policy_response['Policy']['Arn']
)