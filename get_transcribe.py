import datetime 
import os 
import boto3
import time

def upload_file_to_s3(local_file_path, bucket_name, s3_file_name):
    # Create an S3 client
    s3_client = boto3.client('s3')
    # Upload the file to S3
    s3_client.upload_file(local_file_path, bucket_name, s3_file_name)
    print(f"File {local_file_path} uploaded to bucket {bucket_name} as {s3_file_name}.")

def transcribe_audio(file_uri, job_name, language_code='en-US', media_format='mp3'):
    # Create a client for the Amazon Transcribe service
    transcribe_client = boto3.client('transcribe')

    # Start transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        MediaFormat=media_format,
        LanguageCode=language_code
    )

    print(f"Transcription job '{job_name}' started...")

    # Wait for the transcription job to complete, checking the status every 30 seconds
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for transcription to complete...")
        time.sleep(30)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        print(f"Transcription job completed successfully.")
        # Fetch and print the transcription
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        print(f"Transcript URL: {transcription_url}")
        return transcription_url
    else:
        print("Transcription job failed.")
        return None

# Specify your local file path, S3 bucket name, and the name you want the file to have in S3
local_file_path = 'examples/ba_q_3_2023_10_25_earnings_summary.mp3'
bucket_name = 'a204383-scw-use1-athensws'
subfolder_name = 'posture/data/aginw/emotional_detection'

base_name = os.path.basename(local_file_path)
s3_file_name = os.path.join(subfolder_name, base_name)

# Upload the file to S3
upload_file_to_s3(local_file_path, bucket_name, s3_file_name)

# Construct the S3 URI of the uploaded file
file_uri = f's3://{bucket_name}/{s3_file_name}'

# Start the transcription job with the S3 URI
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
job_name = 'emotional_detection_' + base_name.split('.')[0] + now # This must be unique for every job
transcription_result_url = transcribe_audio(file_uri, job_name)

if transcription_result_url:
    # If you need to do something with the transcription result, you can download it from the URL
    print("Transcription completed. Check the URL for results.")
