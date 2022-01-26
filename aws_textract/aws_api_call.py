import io

import boto3
from PIL import Image


class AwsTextExtraction:
    def __init__(self, img):
        self.image = Image.open(img)
        img_format = self.image.format
        imgByteArr = io.BytesIO()
        self.image.save(imgByteArr, format=img_format)
        self.aws_object = boto3.client(aws_access_key_id='AKIA5JG7A3CR64EEXG4V',
                                       aws_secret_access_key="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",
                                       service_name='textract',
                                       region_name='eu-west-1',
                                       endpoint_url='https://textract.eu-west-1.amazonaws.com')
        self.image_bytes = imgByteArr.getvalue()
        self.blocks_detect = self.get_response()

    def get_response(self):
        response_d = self.aws_object.detect_document_text(Document={'Bytes': self.image_bytes})
        return response_d['Blocks']

    def get_raw_text(self):
        text = ""
        for item in self.blocks_detect:
            if item["BlockType"] == "LINE":
                text = text + " " + item["Text"]
        return text
