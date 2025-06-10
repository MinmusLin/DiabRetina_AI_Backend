import os
import uuid
import requests
import io
import base64
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
from flask_cors import CORS
from scipy import ndimage
from dotenv import load_dotenv

load_dotenv()

VOLCENGINE_API_URL = os.getenv('VOLCENGINE_API_URL')
VOLCENGINE_API_KEY = os.getenv('VOLCENGINE_API_KEY')

if not VOLCENGINE_API_URL or not VOLCENGINE_API_KEY:
    raise ValueError('Missing required environment variables: VOLCENGINE_API_URL or VOLCENGINE_API_KEY')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = None

os.makedirs('diagnostic-report', exist_ok=True)
os.makedirs('original-image', exist_ok=True)
os.makedirs('predicted-image', exist_ok=True)
os.makedirs('preprocessed-image', exist_ok=True)

LESION_COLORS = {
    'EX': (255, 0, 96),
    'MA': (0, 223, 162),
    'HE': (0, 121, 255),
    'SE': (246, 250, 112)
}

LESION_TYPES = {
    1: 'EX',
    2: 'HE',
    3: 'MA',
    4: 'SE'
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
    return colored_mask, pred_mask

def count_lesions(pred_mask):
    lesion_counts = {'EX': 0, 'HE': 0, 'MA': 0, 'SE': 0}
    for class_idx, lesion_type in LESION_TYPES.items():
        binary_mask = (pred_mask == class_idx).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        labeled, num_features = ndimage.label(binary_mask)
        for i in range(1, num_features + 1):
            area = np.sum(labeled == i)
            if area >= 10:
                lesion_counts[lesion_type] += 1
    return lesion_counts

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
    colored_mask, pred_mask = create_colored_mask(prediction)
    lesion_counts = count_lesions(pred_mask)
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
        'predicted_image': base64.b64encode(predicted_img).decode('utf-8'),
        'lesion_counts': lesion_counts
    }

def get_severity_text(severity_code):
    severity_map = {
        '0': '健康',
        '1': '轻度非增殖性 DR（Mild-NPDR）',
        '2': '中度非增殖性 DR（Moderate-NPDR）',
        '3': '重度非增殖性 DR（Severe-NPDR）',
        '4': '增殖性 DR（PDR）'
    }
    return severity_map.get(severity_code, '未知')

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

@app.route('/generate_diagnosis', methods=['POST'])
def generate_diagnosis():
    try:
        data = request.json
        required_fields = [
            'name',
            'gender',
            'age',
            'occupation',
            'contact',
            'address',
            'chief_complaint',
            'present_illness',
            'past_history',
            'ma_count',
            'he_count',
            'ex_count',
            'se_count',
            'ma_severity',
            'he_severity',
            'ex_severity',
            'se_severity',
            'clinical_diagnosis',
            'treatment_plan'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        prompt = f'''Prompt 定义：
【角色定义】
你是糖尿病性视网膜病变诊断智能平台的医疗助手，基于循证医学提供疾病知识科普、诊断流程解释和预防建议，不替代专业医疗建议。
【医学背景】
糖尿病性视网膜病变涉及四种病灶：
- 微动脉瘤（Microaneurysm，MA）：视网膜毛细血管壁局部膨出形成的微小瘤状结构，直径较小，是糖尿病视网膜病变最早的病理特征。
- 出血点（Hemorrhage，HE）：视网膜深层毛细血管破裂导致的点状或片状出血，位于内核层或外丛状层。
- 硬性渗出（Hard Exudates，EX）：脂质和蛋白质渗漏沉积于外丛状层，呈蜡黄色点片状，边界清晰，提示慢性视网膜水肿。
- 软性渗出（Soft Exudates，SE）：神经纤维层微梗死导致的轴浆蓄积，呈白色絮状、边界模糊，阻碍下方血管观察。
根据国际临床 DR 严重程度量表，DR 共分为 5 级：健康、轻度非增殖性 DR（Mild non-proliferative DR，Mild-NPDR）、中度非增殖性 DR（Moderate non-proliferative DR，Moderate-NPDR）、重度非增殖性 DR（Severe non-proliferative DR，Severe-NPDR）和增殖性 DR（Proliferative DR，PDR）。
【任务指令】
请你根据患者基本信息、主诉、现病史、既往史、病灶严重程度分级、临床诊断意见和治疗方案等信息，生成 AI 辅助诊断意见。
患者基本信息：
- 姓名：{data['name']}
- 性别：{data['gender']}
- 年龄：{data['age']}
- 职业：{data['occupation']}
- 联系方式：{data['contact']}
- 家庭住址：{data['address']}
主诉：{data['chief_complaint']}
现病史：{data['present_illness']}
既往史：{data['past_history']}
病灶严重程度分级：
- 微动脉瘤（MA）：{data['ma_count']} 处，严重程度：{get_severity_text(data['ma_severity'])}
- 视网膜出血（HE）：{data['he_count']} 处，严重程度：{get_severity_text(data['he_severity'])}
- 视网膜渗出（EX）：{data['ex_count']} 处，严重程度：{get_severity_text(data['ex_severity'])}
- 硬性渗出（SE）：{data['se_count']} 处，严重程度：{get_severity_text(data['se_severity'])}
临床诊断意见：{data['clinical_diagnosis']}
治疗方案：{data['treatment_plan']}
【输出要求】
以无任何文本样式的纯文本格式输出，第一段为患者病史概述，第二段为病灶程度评估，第三段为辅助诊断意见，第四段为治疗方案建议，第五段为注意事项。
【输出格式】
患者病史概述：[简要总结患者主诉、现病史及既往史等]
病灶程度评估：[列出微动脉瘤、出血点、硬性渗出、软性渗出的数量及严重程度等]
辅助诊断意见：[基于病灶评估及病史，给出 AI 支持的 DR 辅助诊断意见等]
治疗方案建议：[提出疾病控制目标、随访周期及必要干预措施，给出 AI 支持的 DR 治疗方案建议等]
注意事项：本回复基于公开医学指南，AI 辅助诊断意见仅供参考，不替代专业医疗建议。请结合临床医生评估制定个性化治疗方案。[禁止修改注意事项]'''
        headers = {
            'Authorization': f'Bearer {VOLCENGINE_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'doubao-1-5-pro-32k-250115',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        response = requests.post(VOLCENGINE_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return jsonify({'ai_response': 'AI 辅助诊断意见生成失败。'})
        response_data = response.json()
        return jsonify({'ai_response': response_data['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)