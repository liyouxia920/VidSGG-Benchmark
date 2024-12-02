import torch
import numpy as np
from functools import reduce
from lib.ults.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

class BasicSceneGraphEvaluator:
    def __init__(self, mode, AG_object_classes, AG_all_predicates, AG_attention_predicates, AG_spatial_predicates, AG_contacting_predicates,
                 iou_threshold=0.3, constraint=False, semithreshold=None):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.constraint = constraint  # Semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semithreshold = semithreshold

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            if len(v) > 0:
                print('R@%i: %f' % (k, np.mean(v)))
            else:
                print('R@%i: N/A' % k)

    def evaluate_scene_graph(self, gt, pred):
        """Collect ground truth and predictions."""
        print(f"Evaluating mode: {self.mode}")
        print(f"Ground truth input: {gt}")
        print(f"Prediction input: {pred}")

        gt_boxes = []
        gt_classes = []
        gt_relations = []
        object_id_offset = 0  # To adjust indices when combining frames

        for frame_idx, frame_gt in enumerate(gt):
            frame_object_idx_mapping = {}  # Mapping from frame-specific index to global index
            person_indices_in_frame = []

            # Collect ground truth boxes and classes
            for ann_idx, ann in enumerate(frame_gt):
                bbox = ann['bbox']
                label = ann['class']

                gt_boxes.append(bbox)
                gt_classes.append(label)
                global_obj_idx = object_id_offset + ann_idx
                frame_object_idx_mapping[ann_idx] = global_obj_idx

                if label == self.AG_object_classes.index('person'):
                    person_indices_in_frame.append(ann_idx)

            # For each object that is not a person, collect relationships
            for ann_idx, ann in enumerate(frame_gt):
                label = ann['class']
                if label == self.AG_object_classes.index('person'):
                    continue  # Skip person annotations

                obj_global_idx = frame_object_idx_mapping[ann_idx]
                for rel_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                    for predicate in ann[rel_type]:
                        for person_ann_idx in person_indices_in_frame:
                            person_global_idx = frame_object_idx_mapping[person_ann_idx]
                            if predicate in self.AG_all_predicates:
                                predicate_idx = self.AG_all_predicates.index(predicate)
                                gt_relations.append([person_global_idx, obj_global_idx, predicate_idx])
                            else:
                                continue

            object_id_offset += len(frame_gt)

        gt_boxes = np.array(gt_boxes)
        gt_classes = np.array(gt_classes)
        gt_relations = np.array(gt_relations)

        print("Ground truth boxes:", gt_boxes)
        print("Ground truth classes:", gt_classes)
        print("Ground truth relations:", gt_relations)

        gt_entry = {
            'gt_classes': gt_classes,
            'gt_relations': gt_relations,
            'gt_boxes': gt_boxes,
        }

        # Construct pred_entry based on mode

        # print("######## Pred ########")
        # print(pred)

        if self.mode == 'predcls':
            pred_entry = {
                'pred_boxes': pred['pred_boxes'],
                'pred_classes': pred['pred_classes'],
                'pred_rel_inds': pred['pred_rel_inds'],
                'rel_scores': pred['rel_scores'],
                'obj_scores': np.ones(len(pred['pred_classes'])),  # All objects are correctly classified
            }
        else:
            pred_entry = {
                'pred_boxes': pred['pred_boxes'],
                'pred_classes': pred['pred_classes'],
                'pred_rel_inds': pred['pred_rel_inds'],
                'rel_scores': pred['rel_scores'],
                'obj_scores': pred['obj_scores'],
            }


        print("Predicted boxes:", pred_entry['pred_boxes'])
        print("Predicted classes:", pred_entry['pred_classes'])
        print("Predicted relations:", pred_entry['pred_rel_inds'])

        evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                           iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semithreshold)


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold=0.9, **kwargs):
    if gt_entry['gt_relations'].size == 0:
        print("No ground truth relations. Skipping relation evaluation.")
        return

    if pred_entry['pred_rel_inds'].size == 0:
        print("No predicted relations. Skipping relation evaluation.")
        return

    print("Evaluating recall...")
    print("Ground truth relations:", gt_entry['gt_relations'])
    print("Predicted relations:", pred_entry['pred_rel_inds'])

    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']

    if method == 'semi':
        pred_rels = []
        for i, j in enumerate(pred_rel_inds):
            for k in np.where(rel_scores[i] > threshold)[0]:
                pred_rels.append(np.append(j, k))
        pred_rels = np.array(pred_rels)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)))

    print("Filtered predicted relations:", pred_rels)

    pred_to_gt, _, _ = evaluate_recall(
        gt_rels, gt_boxes, gt_classes,
        pred_rels, pred_boxes, pred_classes,
        None, None, phrdet=mode == 'phrdet',
        **kwargs)

    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])
        rec_i = float(len(match)) / float(gt_rels.shape[0]) if gt_rels.shape[0] > 0 else 0
        result_dict[mode + '_recall'][k].append(rec_i)

    print(f"Recall results: {result_dict[mode + '_recall']}")

def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.3, phrdet=False):
    """
    Evaluate recall.
    """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)

    pred_triplets, pred_triplet_boxes, _ = _triplet(pred_rels[:, 2],
                                                    pred_rels[:, :2],
                                                    pred_classes,
                                                    pred_boxes)


    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    return pred_to_gt, pred_triplets, None

def _triplet(predicates, relations, classes, boxes):
    """
    Format predictions into triplets.
    """
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))
    return triplets, triplet_boxes, None

def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Compute matches between predicted and ground truth triplets.
    """
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for _ in range(len(pred_triplets))]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match]):
        boxes = pred_boxes[keep_inds]
        sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
        obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt