import cv2
import torch
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from fer import FER
from deepface import DeepFace
import json

# Load the DETR model and processor for object detection
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load the caption generation model and tokenizer
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

# Caption generation settings
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Initialize the emotion detector
emotion_detector = FER(mtcnn=True)

# Function to generate caption for an image
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = caption_model.generate(pixel_values, **gen_kwargs)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Function to process video frames and collect data for report
def process_frame(frame, report_data, frame_idx):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = detr_processor(images=image_rgb, return_tensors="pt")
    outputs = detr_model(**inputs)

    # Object detection
    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = detr_model.config.id2label[label.item()]
        confidence = round(score.item(), 3)

        detected_objects.append((label_name, confidence, box))

        # Draw bounding box
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Put label and confidence score on the bounding box
        cv2.putText(frame, f'{label_name}: {confidence}', (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Collecting interaction data
    for i, (label1, conf1, box1) in enumerate(detected_objects):
        for j, (label2, conf2, box2) in enumerate(detected_objects):
            if i >= j:  # Avoid double counting or self-comparison
                continue

            interaction = predict_interaction(box1, box2)
            if interaction:
                mid_point1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
                mid_point2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
                cv2.line(frame, (int(mid_point1[0]), int(mid_point1[1])), (int(mid_point2[0]), int(mid_point2[1])), (0, 255, 0), 1)
                cv2.putText(frame, interaction, ((int(mid_point1[0]) + int(mid_point2[0])) // 2, (int(mid_point1[1]) + int(mid_point2[1])) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                report_data["interactions"].append({"object1": label1, "object2": label2, "interaction": interaction, "timestamp": frame_idx})

    # Emotion recognition and engagement assessment
    for (label, _, box) in detected_objects:
        if label == "person":
            x1, y1, x2, y2 = map(int, box)
            face = image_rgb[y1:y2, x1:x2]  # Extract the face region
            if face.size > 0:  # Ensure the face region is not empty
                # Recognize emotion using FER
                emotion, _ = emotion_detector.top_emotion(face)
                if emotion is not None:
                    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    report_data["emotions"].append({"emotion": emotion, "timestamp": frame_idx})

                # Assess engagement using DeepFace (using age as a proxy for engagement level)
                try:
                    analysis = DeepFace.analyze(face, actions=['age'], enforce_detection=False)
                    if 'age' in analysis:
                        age = analysis['age']
                        engagement = "high" if age > 25 else "low"
                        cv2.putText(frame, f'Engagement: {engagement}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        report_data["engagement"].append({"engagement": engagement, "timestamp": frame_idx})
                except:
                    pass

    return frame

# Function to predict human-object interaction
def predict_interaction(box1, box2):
    x1_center, y1_center = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2_center, y2_center = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

    distance = np.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)

    if distance < 50:  # Threshold for "near"
        return "near"
    elif y1_center < y2_center:
        return "above"
    else:
        return "below"

    return None

# Function to process video and save output
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Data structure to collect report data
    report_data = {"emotions": [], "engagement": [], "interactions": []}

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image for caption generation
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = generate_caption(frame_image)

        # Process frame for detection, emotion, engagement, and interactions
        processed_frame = process_frame(frame, report_data, frame_idx)

        # Add caption to the frame
        cv2.putText(processed_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(processed_frame)

    cap.release()
    out.release()

    # Generate the report
    generate_report(report_data)

# Function to generate the final report
def generate_report(report_data):
    # Summary of emotions
    emotion_summary = {emotion['emotion']: 0 for emotion in report_data["emotions"]}
    for emotion in report_data["emotions"]:
        emotion_summary[emotion['emotion']] += 1

    # Summary of engagement
    engagement_summary = {engagement['engagement']: 0 for engagement in report_data["engagement"]}
    for engagement in report_data["engagement"]:
        engagement_summary[engagement['engagement']] += 1

    with open("report.txt", "w") as report_file:
        report_file.write("Therapy Session Report\n")
        report_file.write("======================\n\n")

        report_file.write("Emotional Analysis:\n")
        for emotion, count in emotion_summary.items():
            report_file.write(f"- {emotion}: {count} occurrences\n")
        report_file.write("\nDetailed Emotions over Time:\n")
        for emotion in report_data["emotions"]:
            report_file.write(f"- Timestamp: {emotion['timestamp']} - Emotion: {emotion['emotion']}\n")

        report_file.write("\nEngagement Analysis:\n")
        for engagement, count in engagement_summary.items():
            report_file.write(f"- {engagement}: {count} occurrences\n")
        report_file.write("\nDetailed Engagement over Time:\n")
        for engagement in report_data["engagement"]:
            report_file.write(f"- Timestamp: {engagement['timestamp']} - Engagement: {engagement['engagement']}\n")

        report_file.write("\nObject Interactions:\n")
        for interaction in report_data["interactions"]:
            report_file.write(f"- Timestamp: {interaction['timestamp']} - {interaction['object1']} and {interaction['object2']} are {interaction['interaction']}\n")

# Replace "input_video.mp4" with the path to your input video file
process_video("/content/drive/MyDrive/Intensive Therapy Effective For Autism Treatment (online-video-cutter.com) (1).mp4", "output_video.mp4")