# README

1. Download the dataset  [Action Genome](https://github.com/JingweiJ/ActionGenome) to put it in to the folder "action_genome".

   ```
   |-- action_genome
       |-- annotations   #gt annotations
       |-- frames        #sampled frames
       |-- videos        #original videos
   ```

2. Download the pre-trained model weights from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

   ```
   mkdir weights
   cd weights
   wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   cd ..
   ```

3. Thanks to [STTran](https://github.com/yrcong/STTran), in the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the `action_genome/annotations/`

4. Modify the `data_path` of `config.py`

   ```python
   parser.add_argument('-data_path', default='/Users/liyouxia/Dissertation/VidSGG Benchmark/action_genome/', type=str)
   ```

5. ```
   pip install -r requirements.txt
   ```

6. Use the true API Key in `test.py`:

   ```python
   # Replace with your actual API Key
   API_KEY = "API_KEY"  # 请替换为您的实际API密钥
   genai.configure(api_key=API_KEY)
   ```

7. The project Structure:

   ```
   |-- action_genome
       |-- annotations   # gt annotations
           |-- frame_list.txt
           |-- object_bbox_and_relationship_filtersmall.pkl
           |-- object_bbox_and_relationship.pkl
           |-- object_classes.txt
           |-- person_bbox.pkl
           |-- relationship_classes.txt
       |-- frames        # sampled frames
       |-- videos        # original videos
   |-- dataloader
       |-- action_genome.py
   |-- groundingdino
       |-- __init__.py
       |-- config
       |-- datasets
       |-- models
       |-- util
       |-- version.py
   |-- lib
       |-- config.py
       |-- draw_rectangles
       |-- evaluation_recall.py
       |-- fpn
       |-- object_detector.py
       |-- ults
   |-- test.py
   |-- weights
       |-- groundingdino_swint_ogc.pth
   |-- prompt.txt
   |-- Readme.md
   |-- requirements.txt
   ```

8. ```
   python test.py -mode predcls
   python test.py -mode sgcls
   python test.py -mode sgdet
   ```
