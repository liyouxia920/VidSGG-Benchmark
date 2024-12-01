import numpy as np
import torch
import json
from dataloader.action_genome import AG, collate_fn
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import Detector
import warnings
import base64
from io import BytesIO
from torchvision.transforms.functional import to_pil_image

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=4)

# Configure Gemini API
API_KEY = "AIzaSyDmScivUqX91lzHgtJ_DWa3_5xgulAAUWk"  # Replace with your actual API Key

# Initialize Gemini model
def get_gemini_model(api_key, max_output_tokens=8192, model_card="gemini-1.5-flash-latest"):
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel(
        model_card,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

gemini_model = get_gemini_model(API_KEY)

# Uniform sampling function
def sample_frames_uniformly(frame_list, num_samples):
    if len(frame_list) <= num_samples:
        return list(range(len(frame_list)))
    indices = np.linspace(0, len(frame_list) - 1, num_samples, dtype=int)
    return [int(i) for i in indices]  # Ensure indices are int

# Prepare images
def prepare_images(im_data_sampled):
    images_encoded = []
    for frame in im_data_sampled:
        # Convert Tensor to PIL image
        pil_image = to_pil_image(frame.cpu())
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        # Convert to Base64 encoded string
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        images_encoded.append(image_base64)
    return images_encoded

# Generate Prompt
def prepare_prompt(detector_output, sampled_frames, AG_dataset, images_encoded, mode):
    frame_details = []
    global_object_id = 0  # Global object index
    object_id_mapping = {}  # Map global object ID to detector_output index

    for frame_idx, (frame_id, encoded_image) in enumerate(zip(sampled_frames, images_encoded)):
        frame_id = int(frame_id)
        frame_objects = []
        indices_in_frame = (detector_output['im_idx_obj'] == frame_id).nonzero(as_tuple=True)[0]
        for idx in indices_in_frame:
            label = detector_output['labels'][idx]
            bbox = detector_output['boxes'][idx][1:].tolist()  # Exclude image index
            frame_objects.append({
                "id": int(global_object_id),
                "name": AG_dataset.object_classes[int(label.item())],
                "bbox": bbox
            })
            object_id_mapping[int(global_object_id)] = int(idx.item())
            global_object_id += 1
        frame_details.append({
            "frame_id": int(frame_id),
            "image": encoded_image,
            "objects": frame_objects
        })

    # Construct the prompt
    # prompt = "PROMPT_PLACEHOLDER"  # Replace with your actual prompt or keep as a placeholder

        prompt = f"""
    You are provided with a series of video frames and detected objects in each frame. Each frame includes an image and a list of detected objects. Your task is to analyze these frames and generate a detailed scene graph description in JSON format, including objects and their relationships.
    Use the object categories, their spatial positions, and the visual information in the provided images to infer possible relationships. The output must match the input format expected by the evaluation script.

    **Input:**
    - Frames: A list of frames with their IDs, images (Base64-encoded), and detected objects.
    ```json
    {json.dumps(frame_details)}

    	•	Relationships should be in the following format:
    	•	"subject_id": ID of the subject object (from the "id" field)
    	•	"object_id": ID of the object (from the "id" field)
    	•	"predicate": the relationship label (choose from the following categories)
    	•	Attention Relationships: {AG_dataset.attention_relationships}
    	•	Spatial Relationships: {AG_dataset.spatial_relationships}
    	•	Contacting Relationships: {AG_dataset.contacting_relationships}

    Example:
    If a person is holding a broom, the relationship could be {{"subject_id": person_id, "object_id": broom_id, "predicate": "holding"}}.

    Output Format:
    {{
      "objects": [{{ "id": int, "name": str, "bbox": [float, float, float, float] }}],
      "relationships": [{{ "subject_id": int, "object_id": int, "predicate": str }}]
    }}

    """


    # Print the prompt for debugging
    print("##### Gemini Prompt #####")
    print(prompt)

    return prompt, object_id_mapping, frame_details

# Initialize configuration
conf = Config()
for key, value in conf.args.items():
    print(f"{key}: {value}")

# Load dataset
AG_dataset = AG(
    split='test',  # Set the dataset split to 'test'
    datasize=conf.datasize,
    data_path=conf.data_path,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == "predcls" else True
)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=collate_fn)

device = torch.device("cpu")
object_detector = Detector(
    train=False,
    object_classes=AG_dataset.object_classes,
    use_SUPPLY=True,
    mode=conf.mode,
    device=device
).to(device)
object_detector.eval()

# Initialize Evaluators
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint="with"
)

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint="semi", semithreshold=0.9
)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint="no"
)

# Test loop
with torch.no_grad():
    for b, data in enumerate(dataloader):
        im_data = data[0][0]  # [T, C, H, W]
        annotations = data[1][0]  # List of annotations per frame
        indices = data[2]  # Video index
        gt_annotation = AG_dataset.gt_annotations[indices[0]]

        num_frames = len(im_data)
        sampled_frames = sample_frames_uniformly(list(range(num_frames)), num_samples=10)

        # Get sampled frames data
        im_data_sampled = [im_data[i] for i in sampled_frames]
        annotations_sampled = [annotations[i] for i in sampled_frames]

        # Prepare images
        images_encoded = prepare_images(im_data_sampled)

        # Update entry with images and annotations
        entry = object_detector(
            images_list=[im_data_sampled],
            annotations_list=[annotations_sampled],
            indices=indices,
            sampled_frames=sampled_frames  # New parameter
        )

        # Remove image_index column from boxes
        boxes = entry['boxes'][:, 1:]
        if conf.mode == 'predcls':
            labels = entry['labels']
            scores = torch.ones_like(entry['labels'], dtype=torch.float32)
        else:
            labels = entry['pred_labels']
            scores = entry['pred_scores']

        detector_output = {
            'boxes': entry['boxes'],  # Include image index for mapping
            'labels': labels,
            'scores': scores,
            'im_idx_obj': entry['im_idx_obj'],
            'im_idx_pair': entry['im_idx_pair'],
            'pair_idx': entry['pair_idx']
        }

        # print(f"Processing video with indices: {indices}")
        frame_names = [AG_dataset.video_list[indices[0]][i] for i in sampled_frames]
        # print(f"Sampled frame names: {frame_names}")

        # Prepare prompt and object ID mapping
        prompt, object_id_mapping, frame_details = prepare_prompt(
            detector_output, sampled_frames, AG_dataset, images_encoded, conf.mode
        )

        # Prepare Gemini input
        gemini_input = [prompt] + images_encoded  # Gemini input: [text_prompt, image1, image2, ...]

        # Print what is being fed to Gemini
        # print("##### Gemini Input #####")
        # print(gemini_input)

        # Call Gemini API
        try:
            # Assuming Gemini accepts multiple inputs (text and images)
            response = gemini_model.generate_content(gemini_input, stream=False)
            response.resolve()
            res = response.candidates[0].content.parts[0].text
            print(f"Gemini Response: {res}")

        except Exception as e:
            print(f"Gemini Error: {e}")
            continue

        # Parse Gemini's response
        try:
            pred_data = json.loads(res)
            if 'objects' not in pred_data or 'relationships' not in pred_data:
                print("Gemini response does not contain required keys.")
                continue
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            continue

        # Prepare predictions for evaluator
        pred_rel_inds = []
        predicates = []
        pred_boxes = []
        pred_classes = []
        obj_scores = []

        # Map Gemini's object IDs to indices in detector_output
        gemini_obj_id_to_new_idx = {}
        for obj in pred_data['objects']:
            obj_id = obj['id']
            if obj_id in object_id_mapping:
                idx = object_id_mapping[obj_id]
                gemini_obj_id_to_new_idx[obj_id] = len(pred_boxes)  # New index
                pred_boxes.append(detector_output['boxes'][idx][1:].cpu().numpy())  # Exclude image index
                pred_classes.append(detector_output['labels'][idx].item())
                obj_scores.append(detector_output['scores'][idx].item())
            else:
                print(f"Invalid object ID in Gemini output: {obj_id}")

        if len(pred_boxes) == 0:
            print("No valid objects predicted.")
            continue

        # Re-map relationships using the new indices
        for rel in pred_data['relationships']:
            subj_id = rel['subject_id']
            obj_id = rel['object_id']
            predicate_label = rel['predicate']
            if predicate_label in AG_dataset.relationship_classes:
                predicate_idx = AG_dataset.relationship_classes.index(predicate_label)
                if subj_id in gemini_obj_id_to_new_idx and obj_id in gemini_obj_id_to_new_idx:
                    subj_idx = gemini_obj_id_to_new_idx[subj_id]
                    obj_idx = gemini_obj_id_to_new_idx[obj_id]
                    pred_rel_inds.append([subj_idx, obj_idx])
                    predicates.append(predicate_idx)
                else:
                    print(f"Invalid object IDs in relationship: {rel}")
            else:
                print(f"Unknown predicate: {predicate_label}")
                continue

        if len(pred_rel_inds) == 0:
            print("No valid relationships predicted.")
            continue

        pred_entry = {
            'pred_boxes': np.array(pred_boxes),
            'pred_classes': np.array(pred_classes),
            'pred_rel_inds': np.array(pred_rel_inds, dtype=np.int64),
            'obj_scores': np.array(obj_scores),
            'rel_scores': np.zeros((len(pred_rel_inds), len(AG_dataset.relationship_classes)))
        }

        # Assign predicted predicate scores
        for i, predicate_idx in enumerate(predicates):
            pred_entry['rel_scores'][i, :] = 0.01  # Assign a small initial score to all predicates
            pred_entry['rel_scores'][i, predicate_idx] = 1.0  # Assign high score to predicted predicate

        # Print variables for debugging
        # print("########### Pred Entry #############")
        # print(pred_entry)

        # Evaluate
        evaluator1.evaluate_scene_graph(gt_annotation, pred_entry)
        evaluator2.evaluate_scene_graph(gt_annotation, pred_entry)
        evaluator3.evaluate_scene_graph(gt_annotation, pred_entry)

    # Print results

    print("———————––with constraint—————————––")
    evaluator1.print_stats()
    print("———————––semi constraint—————————––")
    evaluator2.print_stats()
    print("———————––no constraint—————————––")
    evaluator3.print_stats()