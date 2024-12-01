import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T

# 导入GroundingDINO所需的模块
from groundingdino.util.inference import load_model, predict
from groundingdino.util.utils import clean_state_dict

class Detector(nn.Module):
    def __init__(self, train, object_classes, use_SUPPLY, mode='predcls', device='cpu'):
        super(Detector, self).__init__()
        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode
        self.device = device

        if mode in ['sgcls', 'sgdet']:
            # 初始化GroundingDINO模型
            model_config_path = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
            model_checkpoint_path = 'weights/groundingdino_swint_ogc.pth'
            self.detection_model = load_model(model_config_path, model_checkpoint_path, device=self.device)
            self.detection_model.eval()

            # 图像预处理变换（如果需要）
            self.transform = T.Compose([
                T.Resize((800, 1333)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def forward(self, images_list, annotations_list, indices, sampled_frames):
        FINAL_BBOXES = []
        FINAL_LABELS = []
        FINAL_SCORES = []
        IMAGE_IDX = []
        HUMAN_IDX = []
        PAIR_IDX = []
        IM_IDX = []

        for idx, (images, annotations) in enumerate(zip(images_list, annotations_list)):
            for frame_idx, (image, anns) in enumerate(zip(images, annotations)):
                frame_id = sampled_frames[frame_idx]  # 使用实际的帧编号
                image_index = frame_id  # 使用实际帧编号作为image_index

                if self.mode == 'predcls':
                    # 使用真实的标签和边界框
                    human_indices_in_frame = []
                    object_indices_in_frame = []

                    for ann_idx, ann in enumerate(anns):
                        bbox = ann['bbox']  # 边界框
                        label = ann['class']

                        FINAL_BBOXES.append(torch.cat([torch.tensor([image_index]), bbox.cpu()]))
                        FINAL_LABELS.append(label)
                        FINAL_SCORES.append(torch.tensor(1.0))  # 置信度得分设为1.0
                        IMAGE_IDX.append(image_index)

                        if label == self.object_classes.index('person'):
                            HUMAN_IDX.append(len(FINAL_LABELS) - 1)
                            human_indices_in_frame.append(len(FINAL_LABELS) - 1)
                        else:
                            object_indices_in_frame.append(len(FINAL_LABELS) - 1)

                    # 生成关系的配对索引
                    for human_idx in human_indices_in_frame:
                        for obj_idx in object_indices_in_frame:
                            PAIR_IDX.append([human_idx, obj_idx])
                            IM_IDX.append(image_index)

                elif self.mode in ['sgcls', 'sgdet']:
                    # 使用GroundingDINO进行对象检测
                    # 将图像移动到设备上并进行预处理
                    transformed_image = image.to(self.device)

                    # 准备文本提示，包含所有对象类别
                    text_prompt = '. '.join(self.object_classes[1:]) + '.'

                    # 使用GroundingDINO进行预测
                    boxes, logits, phrases = predict(
                        model=self.detection_model,
                        image=transformed_image,
                        caption=text_prompt,
                        box_threshold=0.35,
                        text_threshold=0.25,
                        device=self.device
                    )

                    # 处理预测结果
                    boxes = boxes.cpu()
                    # 将边界框从归一化坐标转换为图像坐标
                    boxes[:, 0::2] *= transformed_image.shape[2]  # x坐标
                    boxes[:, 1::2] *= transformed_image.shape[1]  # y坐标

                    human_indices_in_frame = []
                    object_indices_in_frame = []

                    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                        if '(' in phrase:
                            phrase = phrase.split('(')[0].strip()
                        if phrase in self.object_classes:
                            label_idx = self.object_classes.index(phrase)
                            FINAL_BBOXES.append(torch.cat([torch.tensor([image_index]), box]))
                            FINAL_LABELS.append(label_idx)
                            FINAL_SCORES.append(logit)
                            IMAGE_IDX.append(image_index)

                            if phrase == 'person':
                                HUMAN_IDX.append(len(FINAL_LABELS) - 1)
                                human_indices_in_frame.append(len(FINAL_LABELS) - 1)
                            else:
                                object_indices_in_frame.append(len(FINAL_LABELS) - 1)

                    # 生成关系的配对索引
                    for human_idx in human_indices_in_frame:
                        for obj_idx in object_indices_in_frame:
                            PAIR_IDX.append([human_idx, obj_idx])
                            IM_IDX.append(image_index)

        # 将列表转换为张量
        if FINAL_BBOXES:
            FINAL_BBOXES = torch.stack(FINAL_BBOXES)
        else:
            FINAL_BBOXES = torch.empty((0, 5))
        FINAL_LABELS = torch.tensor(FINAL_LABELS, dtype=torch.int64)
        FINAL_SCORES = torch.tensor(FINAL_SCORES)
        IMAGE_IDX = torch.tensor(IMAGE_IDX, dtype=torch.int64)
        HUMAN_IDX = torch.tensor(HUMAN_IDX, dtype=torch.int64)
        PAIR_IDX = torch.tensor(PAIR_IDX, dtype=torch.int64)
        IM_IDX = torch.tensor(IM_IDX, dtype=torch.int64)

        # 准备输出
        entry = {
            'boxes': FINAL_BBOXES.to(self.device),
            'labels': FINAL_LABELS.to(self.device),
            'scores': FINAL_SCORES.to(self.device),
            'im_idx_obj': IMAGE_IDX.to(self.device),
            'im_idx_pair': IM_IDX.to(self.device),
            'human_idx': HUMAN_IDX.to(self.device),
            'pair_idx': PAIR_IDX.to(self.device),
        }

        # 对于'sgcls'和'sgdet'模式，预测标签
        if self.mode in ['sgcls', 'sgdet']:
            entry['pred_labels'] = entry['labels']
            entry['pred_scores'] = entry['scores']

        # 打印检测器的输出用于调试
        # print("########## Detector Entry #########")
        # print(entry)

        return entry