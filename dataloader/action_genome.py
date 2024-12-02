import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

class AG(Dataset):
    def __init__(self, split, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):
        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # Load object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.object_classes.append(line)

        # Manually adjust class names if needed
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # Load relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.relationship_classes.append(line)

        # Manually adjust relationship names if needed
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        self.AG_all_predicates = self.relationship_classes
        print('-------loading annotations---------slowly-----------')

        # Load annotations
        with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
            person_bbox = pickle.load(f)
        with open(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
            object_bbox = pickle.load(f)
        print('--------------------finish!-------------------------')

        # Adjust data size if needed
        if datasize == 'mini':
            person_bbox = {k: person_bbox[k] for k in list(person_bbox.keys())[:80000]}
            object_bbox = {k: object_bbox[k] for k in list(object_bbox.keys())[:80000]}

        # Collect valid frames
        video_dict = {}
        for key in person_bbox.keys():
            if object_bbox[key][0]['metadata']['set'] == split:
                frame_valid = any(obj['visible'] for obj in object_bbox[key])
                if frame_valid:
                    video_name, frame_num = key.split('/')
                    if video_name in video_dict:
                        video_dict[video_name].append(key)
                    else:
                        video_dict[video_name] = [key]

        # For debugging purposes, you can limit the number of videos
        video_dict = dict(list(video_dict.items())[:1])  # Uncomment this line to process all videos

        self.video_list = []
        self.gt_annotations = []
        self.valid_nums = 0
        self.non_gt_human_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0

        # Define image preprocessing transforms
        self.transform = Compose([
            Resize((800, 1333)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        for video_name in video_dict.keys():
            video = []
            gt_annotation_video = []
            for frame_key in video_dict[video_name]:
                if filter_nonperson_box_frame and person_bbox[frame_key]['bbox'].shape[0] == 0:
                    self.non_gt_human_nums += 1
                    continue
                else:
                    video.append(frame_key)
                    self.valid_nums += 1

                # Initialize frame annotations
                gt_annotation_frame = []

                # Process person bounding boxes
                person_bboxes = person_bbox[frame_key]['bbox']
                for person_box in person_bboxes:
                    gt_annotation_frame.append({
                        'bbox': person_box,
                        'class': self.object_classes.index('person'),
                        'attention_relationship': [],
                        'spatial_relationship': [],
                        'contacting_relationship': [],
                        'object_id': len(gt_annotation_frame)  # Assign object ID
                    })

                # Process object bounding boxes and relationships
                obj_list = object_bbox[frame_key]
                for obj in obj_list:
                    if obj['visible']:
                        bbox = [
                            obj['bbox'][0],
                            obj['bbox'][1],
                            obj['bbox'][0] + obj['bbox'][2],
                            obj['bbox'][1] + obj['bbox'][3],
                        ]
                        obj_class = self.object_classes.index(obj['class'])

                        # Collect relationships directly
                        attention_relationship = obj.get('attention_relationship', [])
                        spatial_relationship = obj.get('spatial_relationship', [])
                        contacting_relationship = obj.get('contacting_relationship', [])

                        gt_annotation_frame.append({
                            'bbox': bbox,
                            'class': obj_class,
                            'attention_relationship': attention_relationship,
                            'spatial_relationship': spatial_relationship,
                            'contacting_relationship': contacting_relationship,
                            'object_id': len(gt_annotation_frame)  # Assign object ID
                        })

                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self.video_list.append(video)
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x' * 60)
        if filter_nonperson_box_frame:
            print(f'There are {len(self.video_list)} videos and {self.valid_nums} valid frames')
            print(f'{self.non_person_video} videos are invalid (no person), remove them')
            print(f'{self.one_frame_video} videos are invalid (only one frame), remove them')
            print(f'{self.non_gt_human_nums} frames have no human bbox in GT, remove them!')
        else:
            print(f'There are {len(self.video_list)} videos and {self.valid_nums} valid frames')
            print(f'{self.non_gt_human_nums} frames have no human bbox in GT')
        print('x' * 60)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        frame_names = self.video_list[index]
        processed_ims = []
        annotations = []

        for idx, name in enumerate(frame_names):
            # Load image
            image_path = os.path.join(self.frames_path, name)
            image_pil = Image.open(image_path).convert("RGB")
            orig_W, orig_H = image_pil.size

            # Preprocess image
            im = self.transform(image_pil)
            processed_ims.append(im)

            # Compute scaling factors
            resized_H, resized_W = im.shape[1], im.shape[2]  # Note: im.shape is [C, H, W]
            scale_x = resized_W / orig_W
            scale_y = resized_H / orig_H

            # Load annotations for this frame
            frame_annotations = []
            gt_frame_annotations = self.gt_annotations[index][idx]
            for ann in gt_frame_annotations:
                bbox = torch.tensor(ann['bbox'], dtype=torch.float32)
                # Scale bbox
                bbox[0::2] *= scale_x  # x coordinates
                bbox[1::2] *= scale_y  # y coordinates
                label = ann['class']  # Keep as integer

                # Collect relationships
                attention_relationship = ann['attention_relationship']
                spatial_relationship = ann['spatial_relationship']
                contacting_relationship = ann['contacting_relationship']

                frame_annotations.append({
                    'bbox': bbox,
                    'class': label,
                    'attention_relationship': attention_relationship,
                    'spatial_relationship': spatial_relationship,
                    'contacting_relationship': contacting_relationship
                })
            annotations.append(frame_annotations)

        # Stack images
        images = torch.stack(processed_ims)  # [T, C, H, W]
        return images, annotations, index  # Return index for reference

def collate_fn(batch):
    images = [item[0] for item in batch]  # Each item[0] is a tensor of shape [T, C, H, W]
    annotations = [item[1] for item in batch]  # Each item[1] is a list of frame annotations
    indices = [item[2] for item in batch]
    return images, annotations, indices