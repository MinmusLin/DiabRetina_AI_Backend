import os
import uuid
import numpy as np
import torch
import SimpleITK as sitk
import cv2
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from nets.Transforms import Resize, CenterCrop, ApplyCLAHE, ToTensor
from nets.CAUNet import CAUNet
from flask import Flask, request, jsonify
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None

os.makedirs('original-image', exist_ok=True)
os.makedirs('preprocessed-image', exist_ok=True)
os.makedirs('predicted-image', exist_ok=True)

LESION_COLORS = {
    'EX': (255, 0, 96),
    'MA': (0, 223, 162),
    'HE': (0, 121, 255),
    'SE': (246, 250, 112)
}

model = CAUNet(3, 5)
checkpoint = torch.load('model/model-mcaunet.pth.tar', map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def load_image(image_path):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    transform_origin = transforms.Compose([
        Resize(640),
        CenterCrop(640),
        ApplyCLAHE(green=False)
    ])
    sample_origin = {'image': image_array, 'masks': np.zeros((5, 640, 640))}
    sample_origin = transform_origin(sample_origin)
    transform = transforms.Compose([
        Resize(640),
        CenterCrop(640),
        ApplyCLAHE(green=False),
        ToTensor(green=False)
    ])
    sample = {'image': image_array, 'masks': np.zeros((5, 640, 640))}
    sample = transform(sample)
    return sample['image'].unsqueeze(0), sample_origin['image']

def create_colored_mask(prediction):
    prob = torch.softmax(prediction, dim=1)
    pred_mask = torch.argmax(prob, dim=1).squeeze().cpu().numpy()
    h, w = pred_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colored_mask[pred_mask == 1] = LESION_COLORS['EX']
    colored_mask[pred_mask == 2] = LESION_COLORS['HE']
    colored_mask[pred_mask == 3] = LESION_COLORS['MA']
    colored_mask[pred_mask == 4] = LESION_COLORS['SE']
    return colored_mask

def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.float()
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[-1]
        if output.shape[1] != 5:
            output = output.permute(0, 3, 1, 2)
        output = torch.softmax(output, dim=1)
    return output

def process_image(image_path, file_uuid):
    image_tensor, original_img = load_image(image_path)
    image_tensor = image_tensor.to(device)
    prediction = predict(model, image_tensor)
    colored_mask = create_colored_mask(prediction)
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu().numpy()
    if original_img.shape[0] != 640 or original_img.shape[1] != 640:
        original_img = cv2.resize(original_img, (640, 640))
    if len(original_img.shape) == 2:
        original_img = np.stack([original_img]*3, axis=-1)
    elif original_img.shape[2] == 1:
        original_img = np.repeat(original_img, 3, axis=2)
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)
    else:
        original_img = original_img.astype(np.uint8)
    overlay = original_img.copy()
    mask = (colored_mask != 0).any(axis=2)
    overlay[mask] = cv2.addWeighted(original_img[mask], 0.5, colored_mask[mask], 0.5, 0)
    preprocessed_path = os.path.join('preprocessed-image', f'{file_uuid}.jpg')
    predicted_path = os.path.join('predicted-image', f'{file_uuid}.jpg')
    Image.fromarray(original_img).save(preprocessed_path)
    Image.fromarray(overlay).save(predicted_path)
    img_byte_arr = io.BytesIO()
    Image.fromarray(original_img).save(img_byte_arr, format='JPEG')
    preprocessed_img = img_byte_arr.getvalue()
    img_byte_arr = io.BytesIO()
    Image.fromarray(overlay).save(img_byte_arr, format='JPEG')
    predicted_img = img_byte_arr.getvalue()
    return {
        'uuid': file_uuid,
        'preprocessed_image': base64.b64encode(preprocessed_img).decode('utf-8'),
        'predicted_image': base64.b64encode(predicted_img).decode('utf-8')
    }

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_uuid = str(uuid.uuid4())
    filename = f'{file_uuid}.jpg'
    filepath = os.path.join('original-image', filename)
    try:
        file.save(filepath)
        result = process_image(filepath, file_uuid)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)