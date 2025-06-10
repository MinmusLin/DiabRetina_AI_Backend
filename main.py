import os
import uuid
import requests
import io
import base64
import numpy as np
import torch
import SimpleITK as sitk
import cv2
from datetime import datetime, timedelta
from PIL import Image as PILImage
from torchvision import transforms
from collections import OrderedDict
from nets.Transforms import Resize, CenterCrop, ApplyCLAHE, ToTensor
from nets.CAUNet import CAUNet
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from scipy import ndimage
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect

load_dotenv()

VOLCENGINE_API_URL = os.getenv('VOLCENGINE_API_URL')
VOLCENGINE_API_KEY = os.getenv('VOLCENGINE_API_KEY')

if not VOLCENGINE_API_URL or not VOLCENGINE_API_KEY:
    raise ValueError('Missing required environment variables: VOLCENGINE_API_URL or VOLCENGINE_API_KEY')

pdfmetrics.registerFont(
    TTFont('wqy-zenhei', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
)

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
    PILImage.fromarray(original_img).save(preprocessed_path)
    PILImage.fromarray(overlay).save(predicted_path)
    img_byte_arr = io.BytesIO()
    PILImage.fromarray(original_img).save(img_byte_arr, format='JPEG')
    preprocessed_img = img_byte_arr.getvalue()
    img_byte_arr = io.BytesIO()
    PILImage.fromarray(overlay).save(img_byte_arr, format='JPEG')
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

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        required_fields = [
            'uuid',
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
            'treatment_plan',
            'ai_diagnosis'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        pdf_path = os.path.join('diagnostic-report', f"{data['uuid']}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', fontName='wqy-zenhei', alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='Left', fontName='wqy-zenhei', alignment=TA_LEFT))
        styles.add(ParagraphStyle(name='Comment', fontName='wqy-zenhei', fontSize=9, alignment=TA_LEFT))
        styles['Title'].fontSize = 18
        styles['Title'].alignment = TA_CENTER
        styles['Title'].spaceAfter = 20
        styles['Title'].fontName = 'wqy-zenhei'
        styles.add(ParagraphStyle(name='Subtitle', fontName='wqy-zenhei', fontSize=10, alignment=TA_CENTER, spaceAfter=10))
        styles.add(ParagraphStyle(name='Section', fontName='wqy-zenhei', fontSize=12, alignment=TA_LEFT, spaceBefore=10, spaceAfter=5))
        story = []
        story.append(Paragraph('糖尿病性视网膜病变诊断分析报告', styles['Title']))
        story.append(Paragraph(f'本报告由糖尿病性视网膜病变诊断智能平台 DiabRetina AI 生成', styles['Subtitle']))
        story.append(Spacer(1, 16))
        story.append(Paragraph(f"报告编号：{data['uuid']}", styles['Subtitle']))
        now = datetime.now()
        beijing_time = now + timedelta(hours=8)
        story.append(Paragraph(f"生成时间：{beijing_time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Subtitle']))
        story.append(Spacer(1, 40))
        story.append(Paragraph('患者基本信息', styles['Section']))
        story.append(Spacer(1, 12))
        patient_data = [
            ['姓名', data['name']],
            ['性别', data['gender']],
            ['年龄', data['age']],
            ['职业', data['occupation']],
            ['联系方式', data['contact']],
            ['家庭住址', data['address']]
        ]
        patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'wqy-zenhei'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 24))
        story.append(Paragraph('眼底图像', styles['Section']))
        story.append(Spacer(1, 12))
        original_img_path = os.path.join('preprocessed-image', f"{data['uuid']}.jpg")
        predicted_img_path = os.path.join('predicted-image', f"{data['uuid']}.jpg")
        img_table_data = [
            [Paragraph('眼底原始图像', styles['Center']), Paragraph('眼底检测图像', styles['Center'])],
            [Image(original_img_path, width=3*inch, height=3*inch), Image(predicted_img_path, width=3*inch, height=3*inch)]
        ]
        img_table = Table(img_table_data, colWidths=[3.5*inch, 3.5*inch])
        img_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'wqy-zenhei'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEADING', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('TOPPADDING', (0, 0), (-1, 0), 5),
            ('BOTTOMPADDING', (1, 0), (1, 0), 0)
        ]))
        story.append(img_table)
        story.append(Spacer(1, 24))
        story.append(Paragraph('病灶类型说明', styles['Section']))
        story.append(Spacer(1, 12))
        lesion_colors = {
            'MA': (0, 223, 162),
            'HE': (0, 121, 255),
            'EX': (255, 0, 96),
            'SE': (246, 250, 112)
        }
        lesion_descriptions = [
            ('微动脉瘤（Microaneurysm，MA）', '视网膜毛细血管壁局部膨出形成的微小瘤状结构，是糖尿病视网膜病变最早的病理特征。'),
            ('出血点（Hemorrhage，HE）', '视网膜深层毛细血管破裂导致的点状或片状出血，位于内核层或外丛状层。'),
            ('硬性渗出（Hard Exudates，EX）', '脂质和蛋白质渗漏沉积于外丛状层，呈蜡黄色点片状，边界清晰，提示慢性视网膜水肿。'),
            ('软性渗出（Soft Exudates，SE）', '神经纤维层微梗死导致的轴浆蓄积，呈白色絮状、边界模糊，阻碍下方血管观察。')
        ]
        lesion_explanation_data = []
        for i, (title, desc) in enumerate(lesion_descriptions):
            lesion_type = title.split('（')[-1].split('）')[0].split('，')[-1]
            color = lesion_colors[lesion_type]
            d = Drawing(18, 18)
            d.add(Rect(
                0, 0, 20, 20,
                fillColor=colors.Color(color[0]/255, color[1]/255, color[2]/255),
                strokeColor=colors.lightgrey,
                strokeWidth=1
            ))
            lesion_explanation_data.append([d, Paragraph(f'{title}\n{desc}', styles['Comment'])])
        lesion_explanation_table = Table(lesion_explanation_data, colWidths=[0.5*inch, 5.5*inch])
        lesion_explanation_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'wqy-zenhei'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0, colors.white),
            ('BOX', (0, 0), (-1, -1), 0, colors.white)
        ]))
        story.append(lesion_explanation_table)
        story.append(Spacer(1, 24))
        story.append(Paragraph('患者病史信息', styles['Section']))
        story.append(Spacer(1, 12))
        story.append(Paragraph('主诉：', styles['Left']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(data['chief_complaint'], styles['Left']))
        story.append(Spacer(1, 8))
        story.append(Paragraph('现病史：', styles['Left']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(data['present_illness'], styles['Left']))
        story.append(Spacer(1, 8))
        story.append(Paragraph('既往史：', styles['Left']))
        story.append(Spacer(1, 4))
        story.append(Paragraph(data['past_history'], styles['Left']))
        story.append(Spacer(1, 24))
        story.append(Paragraph('病灶严重程度分级', styles['Section']))
        story.append(Spacer(1, 12))
        lesion_data = [
            ['病灶类型', '数量', '严重程度分级'],
            ['微动脉瘤（MA）', data['ma_count'], get_severity_text(data['ma_severity'])],
            ['出血点（HE）', data['he_count'], get_severity_text(data['he_severity'])],
            ['硬性渗出（EX）', data['ex_count'], get_severity_text(data['ex_severity'])],
            ['软性渗出（SE）', data['se_count'], get_severity_text(data['se_severity'])]
        ]
        lesion_table = Table(lesion_data, colWidths=[1.5*inch, 1*inch, 3*inch])
        lesion_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'wqy-zenhei'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
        ]))
        story.append(lesion_table)
        story.append(Spacer(1, 8))
        story.append(Paragraph('根据国际临床 DR 严重程度量表，DR 共分为 5 级：健康、轻度非增殖性 DR（Mild non-proliferative DR，Mild-NPDR）、中度非增殖性 DR（Moderate non-proliferative DR，Moderate-NPDR）、重度非增殖性 DR（Severe non-proliferative DR，Severe-NPDR）和增殖性 DR（Proliferative DR，PDR）。', styles['Comment']))
        story.append(Spacer(1, 24))
        story.append(Paragraph('临床诊断意见', styles['Section']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(data['clinical_diagnosis'], styles['Left']))
        story.append(Spacer(1, 24))
        story.append(Paragraph('治疗方案', styles['Section']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(data['treatment_plan'], styles['Left']))
        story.append(Spacer(1, 24))
        story.append(Paragraph('AI 辅助诊断意见', styles['Section']))
        story.append(Spacer(1, 12))
        ai_diagnosis_paragraphs = data['ai_diagnosis'].split('\n')
        for para in ai_diagnosis_paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), styles['Left']))
                story.append(Spacer(1, 5))
        story.append(Spacer(1, 48))
        pdf_link = f"http://110.42.214.164:8005/diagnostic-report/{data['uuid']}"
        story.append(Paragraph(f'报告下载链接：{pdf_link}', ParagraphStyle(name='Footer', fontName='wqy-zenhei', fontSize=9, textColor=colors.grey)))
        doc.build(story)
        return jsonify({'report_path': pdf_link})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diagnostic-report/<uuid>', methods=['GET'])
def get_report(uuid):
    try:
        pdf_path = os.path.join('diagnostic-report', f'{uuid}.pdf')
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'Diagnostic report not found'}), 404
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f'diagnostic-report-{uuid}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)