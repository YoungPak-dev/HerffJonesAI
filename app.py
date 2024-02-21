from flask import Flask, request, jsonify, send_file, render_template
import boto3
from PIL import Image, ImageDraw
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Constants for allowed image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# AWS Configuration from Environment Variables
region_name = 'us-east-1'
s3_bucket_name = 'rekognition-custom-projects-us-east-2-952624809b'

# Initialize AWS services
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=region_name)
rekognition = boto3.client('rekognition', aws_access_key_id=aws_access_key_id,
                           aws_secret_access_key=aws_secret_access_key,
                           region_name=region_name)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_collection_id(school_name):
    """Generate the collection ID based on the school name."""
    return f"{school_name}_Collection"

def analyze_class_photo(school_name, photo_bytes):
    """Analyze the uploaded class photo using AWS Rekognition."""
    collection_id = get_collection_id(school_name)
    response = rekognition.search_faces_by_image(
        CollectionId=collection_id,
        Image={'Bytes': photo_bytes},
        FaceMatchThreshold=85,
        MaxFaces=100
    )
    return response['FaceMatches']

def draw_bounding_boxes_and_labels(image, matches):
    """Draw bounding boxes and labels on the image for each matched face."""
    draw = ImageDraw.Draw(image)
    for match in matches:
        box = match['Face']['BoundingBox']
        left = image.width * box['Left']
        top = image.height * box['Top']
        width = image.width * box['Width']
        height = image.height * box['Height']
        
        draw.rectangle([left, top, left + width, top + height], outline="red", width=2)
        name = match['Face']['ExternalImageId'] if 'ExternalImageId' in match['Face'] else "Unknown"
        draw.text((left, top - 10), name, fill="red")

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    school_name = request.form['school_name']
    photo = request.files['photo']
    
    if photo and allowed_file(photo.filename):
        filename = secure_filename(photo.filename)
        photo_path = os.path.join('temp', filename)
        
        # Ensure temp directory exists
        if not os.path.exists('temp'):
            os.makedirs('temp')
        
        photo.save(photo_path)
        
        # Open and process the photo
        with open(photo_path, 'rb') as photo_file:
            photo_bytes = photo_file.read()
            matches = analyze_class_photo(school_name, photo_bytes)
            image = Image.open(photo_path)
            processed_image = draw_bounding_boxes_and_labels(image, matches)
            
            # Prepare processed image for sending
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Cleanup: Remove the saved photo after processing
            os.remove(photo_path)
            
            return send_file(img_byte_arr, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Invalid file type or no file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
