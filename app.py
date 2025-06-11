import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import io

import cv2

# --- Model Configurations (MUST MATCH 9th EXPERIMENT) ---
NUM_POINTS_FULL_GRAPH = 300
NUM_STRIPS = 3
MODEL_NUM_POINTS_ARG = 100  # Hardcoded for robustness
STRIP_WIDTH = 224
STRIP_HEIGHT = 224
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
COORD_DIM = 2
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights

class ViTGraphModel(nn.Module):
    def __init__(self, num_points, vit_model_name, coord_dim, rnn_hidden_size, rnn_num_layers):
        super().__init__()
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.rnn = nn.LSTM(input_size=coord_dim, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)
        self.attention = Attention(encoder_dim=self.vit.config.hidden_size, decoder_dim=rnn_hidden_size)
        self.init_hidden_proj = nn.Linear(self.vit.config.hidden_size, rnn_hidden_size)
        self.fc_out = nn.Linear(rnn_hidden_size + self.vit.config.hidden_size, coord_dim)

    def forward(self, image_pixel_values, first_coord_input):
        B = image_pixel_values.size(0)
        device = image_pixel_values.device
        encoder_outputs = self.vit(pixel_values=image_pixel_values).last_hidden_state
        global_feature = encoder_outputs.mean(dim=1)
        h_0 = self.init_hidden_proj(global_feature).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0).to(device)
        rnn_state = (h_0, c_0)
        current_input_coord = first_coord_input
        outputs = []
        for t in range(self.num_points - 1):
            rnn_input = current_input_coord.unsqueeze(1)
            rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
            rnn_hidden = rnn_output.squeeze(1)
            context_vector, _ = self.attention(rnn_hidden, encoder_outputs)
            combined = torch.cat((rnn_hidden, context_vector), dim=1)
            pred_next_coord = self.fc_out(combined)
            outputs.append(pred_next_coord)
            current_input_coord = pred_next_coord.detach()
        if not outputs:
            return torch.empty(B, 0, self.coord_dim, device=device)
        return torch.stack(outputs, dim=1)

@st.cache_resource
def load_model_and_processor(model_path, vit_name):
    processor = ViTImageProcessor.from_pretrained(vit_name)
    model = ViTGraphModel(
        num_points=100,
        vit_model_name=vit_name,
        coord_dim=COORD_DIM,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS
    )
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model, processor

def download_if_needed(gdrive_url, output_path):
    if not os.path.exists(output_path):
        file_id = None
        if "id=" in gdrive_url:
            file_id = gdrive_url.split("id=")[1].split("&")[0]
        elif "/file/d/" in gdrive_url:
            file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
        elif "open?id=" in gdrive_url:
            file_id = gdrive_url.split("open?id=")[1].split("&")[0]
        if not file_id:
            st.error("Could not parse Google Drive file ID from URL.")
            return False
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(id=file_id, output=output_path, quiet=False)
        st.success("Model downloaded successfully!")
    return True

# --- IMAGE PRE-PROCESSING PIPELINE FUNCTIONS ---
def pil_to_cv2(img):
    img = img.convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def crop_to_plot_area(pil_img):
    cv2_img = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img
    max_rect = None
    max_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > max_area and w > 50 and h > 50:
            max_area = area
            max_rect = (x, y, w, h)
    if max_rect is not None:
        x, y, w, h = max_rect
        cropped = cv2_img[y:y+h, x:x+w]
        return cv2_to_pil(cropped)
    else:
        return pil_img

def isolate_function_line(pil_img):
    cv2_img = pil_to_cv2(pil_img)
    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
    masks = []
    masks.append(cv2.inRange(hsv, (0,70,50), (10,255,255)))
    masks.append(cv2.inRange(hsv, (170,70,50), (180,255,255)))
    masks.append(cv2.inRange(hsv, (90, 60, 0), (130, 255, 255)))
    masks.append(cv2.inRange(hsv, (36, 25, 25), (86, 255,255)))
    color_mask = sum(masks)
    color_mask = cv2.medianBlur(color_mask, 5)
    used_mask = None
    if cv2.countNonZero(color_mask) > 100:
        used_mask = color_mask
    else:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,8)
        used_mask = bw
    # --- Robust connect-the-dots with normalization ---
    contours, _ = cv2.findContours(used_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    if len(centroids) == 0:
        h, w = used_mask.shape
        return cv2_to_pil(np.ones((h, w, 3), dtype=np.uint8) * 255)
    centroids.sort(key=lambda pt: pt[0])
    xs = [pt[0] for pt in centroids]
    ys = [pt[1] for pt in centroids]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min: x_max = x_min + 1
    if y_max == y_min: y_max = y_min + 1
    normalized_points = [((px - x_min) / (x_max - x_min), (py - y_min) / (y_max - y_min)) for px, py in centroids]
    all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours if len(cnt) >= 1])
    if all_points.shape[0] == 0:
        h, w = used_mask.shape
        return cv2_to_pil(np.ones((h, w, 3), dtype=np.uint8) * 255)
    bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(all_points)
    bb_w = max(bb_w, 2)
    bb_h = max(bb_h, 2)
    canvas = np.ones((bb_h, bb_w, 3), dtype=np.uint8) * 255
    scaled_points_for_drawing = [
        (int(px_norm * (bb_w - 1)), int(py_norm * (bb_h - 1)))
        for (px_norm, py_norm) in normalized_points
    ]
    if len(scaled_points_for_drawing) >= 2:
        pts = np.array(scaled_points_for_drawing, dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0,0,0), thickness=3)
    elif len(scaled_points_for_drawing) == 1:
        cv2.circle(canvas, scaled_points_for_drawing[0], radius=1, color=(0,0,0), thickness=2)
    return cv2_to_pil(canvas)

def prepare_image_for_model(pil_img):
    orig_width, orig_height = pil_img.size
    target_h = 224
    resized_w = int(orig_width * (target_h / orig_height))
    img_resized = pil_img.resize((resized_w, target_h), Image.LANCZOS)
    canvas = Image.new('RGB', (672, 224), (255,255,255))
    canvas.paste(img_resized, (0,0))
    width_ratio = resized_w / 672.0
    return canvas, width_ratio

# --- Streamlit App UI ---

st.title("mCGE.AI in action!")

MODEL_GDRIVE_URL = "https://drive.google.com/file/d/1FD3pjwyKa6sK7E_HvO4cU1BwWdBkcjOS/view?usp=drive_link"
LOCAL_MODEL_PATH = "vit_rnn_attn_model_tile_final.pth"
VIT_NAME = VIT_MODEL_NAME

if not download_if_needed(MODEL_GDRIVE_URL, LOCAL_MODEL_PATH):
    st.stop()

try:
    model, processor = load_model_and_processor(LOCAL_MODEL_PATH, VIT_NAME)
except Exception as e:
    st.error(f"Error loading model or processor from {LOCAL_MODEL_PATH}: {e}")
    st.stop()

# --- UI: Input sidebar for first point and chart axes min/max ---
st.sidebar.header("First Point (P0) Input")
p0_y = st.sidebar.number_input("First Point Y (Global, [0,1])", min_value=0.0, max_value=1.0, value=0.5, step=0.000001, format="%.6f")
p0_x = 0.0
st.sidebar.header("Original Chart Axes Range")
user_x_min = st.sidebar.number_input("X-Min", value=0.0, format="%.6f")
user_x_max = st.sidebar.number_input("X-Max", value=1.0, format="%.6f")
user_y_min = st.sidebar.number_input("Y-Min", value=0.0, format="%.6f")
user_y_max = st.sidebar.number_input("Y-Max", value=1.0, format="%.6f")

IMAGE_DIR = "Example Image"
example_images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)
uploaded_file = st.file_uploader("Or upload your own chart image", type=["png", "jpg", "jpeg"])

prepared_img = None
width_ratio = None
pil_img = None  # For example images

if uploaded_file is not None:
    try:
        user_img = Image.open(uploaded_file).convert("RGB")
        st.session_state['original_size'] = user_img.size
        cropped = crop_to_plot_area(user_img)
        line_isolated = isolate_function_line(cropped)
        prepared_img, width_ratio = prepare_image_for_model(line_isolated)
        st.session_state['prepared_img'] = prepared_img
        st.session_state['width_ratio'] = width_ratio
        cols = st.columns(2)
        cols[0].image(user_img, caption="Original uploaded image", channels="RGB")
        cols[1].image(prepared_img, caption=f"Image prepared for AI (672x224, width_ratio={width_ratio:.3f})", channels="RGB")
    except Exception as e:
        st.error(f"Failed to preprocess uploaded image: {e}")
        prepared_img = None
elif example_images:
    selected_image_name = st.selectbox("Select an example wide graph image:", example_images)
    selected_image_path = os.path.join(IMAGE_DIR, selected_image_name)
    pil_img = Image.open(selected_image_path).convert("RGB")
    st.image(pil_img, caption=f"Selected Image: {selected_image_name}", channels="RGB")
else:
    st.error(f"No images found in the folder '{IMAGE_DIR}'. Please add images or upload one.")

do_predict = st.button("Run Prediction")

if do_predict:
    # --- Path 1: User-uploaded image, use full pipeline and denormalize ---
    if uploaded_file is not None and 'prepared_img' in st.session_state and 'width_ratio' in st.session_state:
        img_for_model = st.session_state['prepared_img']
        width_ratio = st.session_state['width_ratio']
        with torch.no_grad():
            current_strip_start_coord_global = np.array([p0_x, p0_y], dtype=np.float32)
            all_global_points = []
            for strip_idx in range(NUM_STRIPS):
                left = strip_idx * STRIP_WIDTH
                upper = 0
                right = left + STRIP_WIDTH
                lower = upper + STRIP_HEIGHT
                strip_img = img_for_model.crop((left, upper, right, lower))
                strip_pixel_values = processor(images=strip_img, return_tensors="pt")['pixel_values'].to(DEVICE)
                strip_x_min = strip_idx / NUM_STRIPS
                local_x = (current_strip_start_coord_global[0] - strip_x_min) * NUM_STRIPS
                local_x = float(np.clip(local_x, 0.0, 1.0))
                local_y = float(np.clip(current_strip_start_coord_global[1], 0.0, 1.0))
                first_coord_for_model_input = torch.tensor([[local_x, local_y]], dtype=torch.float32, device=DEVICE)
                pred_strip_coords = model(strip_pixel_values, first_coord_for_model_input)
                pred_strip_coords = pred_strip_coords.squeeze(0).cpu().numpy()
                strip_points = np.vstack([np.array([[local_x, local_y]], dtype=np.float32), pred_strip_coords])
                global_xs = strip_points[:, 0] / NUM_STRIPS + strip_x_min
                global_ys = strip_points[:, 1]
                global_points = np.stack([global_xs, global_ys], axis=1)
                if strip_idx == 0:
                    all_global_points.append(global_points)
                else:
                    all_global_points.append(global_points[1:])
                current_strip_start_coord_global = global_points[-1]
            coords = np.vstack(all_global_points)
            coords = coords[coords[:,0] <= width_ratio]
            x_real = coords[:,0] / width_ratio * (user_x_max - user_x_min) + user_x_min
            y_real = coords[:,1] * (user_y_max - user_y_min) + user_y_min
            denorm_coords = np.stack([x_real, y_real], axis=1)
        # Dynamic figsize
        if 'original_size' in st.session_state:
            original_width, original_height = st.session_state['original_size']
            aspect_ratio = original_width / original_height
            plot_height = 5
            plot_width = plot_height * aspect_ratio
            final_figsize = (plot_width, plot_height)
        else:
            final_figsize = (12, 4)
        fig, ax = plt.subplots(figsize=final_figsize)
        ax.plot(denorm_coords[:, 0], denorm_coords[:, 1], marker='o', markersize=2)
        ax.set_xlim([user_x_min, user_x_max])
        ax.set_ylim([user_y_min, user_y_max])
        ax.set_aspect('auto')
        ax.set_title("Predicted Graph Points (real-world units)")
        ax.set_xlabel("X (user units)")
        ax.set_ylabel("Y (user units)")
        st.pyplot(fig)
        csv_buffer = io.StringIO()
        np.savetxt(csv_buffer, denorm_coords, delimiter=",", header="x,y", comments="", fmt="%.6f")
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="Download Predicted Coordinates as CSV",
            data=csv_data,
            file_name="predicted_coords.csv",
            mime="text/csv"
        )
    # --- Path 2: Example image, no pre-processing, no denormalization (v5 logic) ---
    elif uploaded_file is None and pil_img is not None:
        img_for_model = pil_img
        with torch.no_grad():
            current_strip_start_coord_global = np.array([0.0, p0_y], dtype=np.float32)
            all_global_points = []
            for strip_idx in range(NUM_STRIPS):
                left = strip_idx * STRIP_WIDTH
                upper = 0
                right = left + STRIP_WIDTH
                lower = upper + STRIP_HEIGHT
                strip_img = img_for_model.crop((left, upper, right, lower))
                strip_pixel_values = processor(images=strip_img, return_tensors="pt")['pixel_values'].to(DEVICE)
                strip_x_min = strip_idx / NUM_STRIPS
                local_x = (current_strip_start_coord_global[0] - strip_x_min) * NUM_STRIPS
                local_x = float(np.clip(local_x, 0.0, 1.0))
                local_y = float(np.clip(current_strip_start_coord_global[1], 0.0, 1.0))
                first_coord_for_model_input = torch.tensor([[local_x, local_y]], dtype=torch.float32, device=DEVICE)
                pred_strip_coords = model(strip_pixel_values, first_coord_for_model_input)
                pred_strip_coords = pred_strip_coords.squeeze(0).cpu().numpy()
                strip_points = np.vstack([np.array([[local_x, local_y]], dtype=np.float32), pred_strip_coords])
                global_xs = strip_points[:, 0] / NUM_STRIPS + strip_x_min
                global_ys = strip_points[:, 1]
                global_points = np.stack([global_xs, global_ys], axis=1)
                if strip_idx == 0:
                    all_global_points.append(global_points)
                else:
                    all_global_points.append(global_points[1:])
                current_strip_start_coord_global = global_points[-1]
            coords = np.vstack(all_global_points)
        # Use the image aspect ratio for the plot
        example_width, example_height = img_for_model.size
        aspect_ratio = example_width / example_height
        plot_height = 5
        plot_width = plot_height * aspect_ratio
        final_figsize = (plot_width, plot_height)
        fig, ax = plt.subplots(figsize=final_figsize)
        ax.plot(coords[:, 0], coords[:, 1], marker='o', markersize=2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('auto')
        ax.set_title("Predicted Graph Points (normalized)")
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")
        st.pyplot(fig)
        csv_buffer = io.StringIO()
        np.savetxt(csv_buffer, coords, delimiter=",", header="x,y", comments="", fmt="%.6f")
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="Download Predicted Coordinates as CSV",
            data=csv_data,
            file_name="predicted_coords.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please upload an image or select an example image.")
