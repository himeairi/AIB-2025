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

# --- Model Configurations (MUST MATCH 9th EXPERIMENT) ---
NUM_POINTS_FULL_GRAPH = 300
NUM_STRIPS = 3
NUM_POINTS_PER_STRIP = NUM_POINTS_FULL_GRAPH // NUM_STRIPS  # 100
MODEL_NUM_POINTS_ARG = NUM_POINTS_PER_STRIP
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
        num_points=MODEL_NUM_POINTS_ARG,
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

st.title("ViTGraphModel Streamlit App (9th Experiment Replication)")

MODEL_GDRIVE_URL = "https://drive.google.com/file/d/1-_n09taoCpOHCggl3YMkdLSL9vBYIWos/view?usp=drive_link"
LOCAL_MODEL_PATH = "vit_rnn_attn_model_tile_final.pth"
VIT_NAME = VIT_MODEL_NAME

if not download_if_needed(MODEL_GDRIVE_URL, LOCAL_MODEL_PATH):
    st.stop()

try:
    model, processor = load_model_and_processor(LOCAL_MODEL_PATH, VIT_NAME)
except Exception as e:
    st.error(f"Error loading model or processor from {LOCAL_MODEL_PATH}: {e}")
    st.stop()

IMAGE_DIR = "Example Image"
example_images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
if not example_images:
    st.error(f"No images found in the folder '{IMAGE_DIR}'. Please add images.")
    st.stop()

selected_image_name = st.selectbox("Select an example wide graph image:", example_images)
selected_image_path = os.path.join(IMAGE_DIR, selected_image_name)

pil_img = Image.open(selected_image_path).convert("RGB")
st.image(pil_img, caption=f"Selected Image: {selected_image_name}", channels="RGB")
img_width, img_height = pil_img.size
expected_width = STRIP_WIDTH * NUM_STRIPS
if img_width != expected_width or img_height != STRIP_HEIGHT:
    st.error(f"Image size must be {STRIP_HEIGHT}px high by {expected_width}px wide (e.g., 224x672 for 3 strips).")
    st.stop()

st.sidebar.header("First Point (P0) Input")
# Allow up to 6 digits after the decimal point
p0_x = st.sidebar.number_input("Global X [0,1]", min_value=0.0, max_value=1.0, value=0.0, step=0.000001, format="%.6f")
p0_y = st.sidebar.number_input("Global Y [0,1]", min_value=0.0, max_value=1.0, value=0.5, step=0.000001, format="%.6f")

do_predict = st.button("Run Prediction")

if do_predict:
    with torch.no_grad():
        current_strip_start_coord_global = np.array([p0_x, p0_y], dtype=np.float32)
        all_global_points = []
        for strip_idx in range(NUM_STRIPS):
            left = strip_idx * STRIP_WIDTH
            upper = 0
            right = left + STRIP_WIDTH
            lower = upper + STRIP_HEIGHT
            strip_img = pil_img.crop((left, upper, right, lower))

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

        final_all_coords_np = np.vstack(all_global_points)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(final_all_coords_np[:, 0], final_all_coords_np[:, 1], marker='o', markersize=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.set_title("Predicted Graph Points")
    ax.set_xlabel("X (global, normalized)")
    ax.set_ylabel("Y (global, normalized)")
    st.pyplot(fig)

    # --- Download CSV Button (no scientific notation, 6 decimals) ---
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, final_all_coords_np, delimiter=",", header="x,y", comments="", fmt="%.6f")
    csv_data = csv_buffer.getvalue()
    st.download_button(
        label="Download Predicted Coordinates as CSV",
        data=csv_data,
        file_name="predicted_coords.csv",
        mime="text/csv"
    )