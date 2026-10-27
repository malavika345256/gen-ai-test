
import io

import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Transformation Studio",
    page_icon="🎨",
    layout="wide"
)


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def image_to_bytes(image):
    """Convert a PIL image into downloadable PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def calculate_mse(original, transformed):
    """Calculate MSE and normalized MSE between two images."""
    original_array = np.asarray(original, dtype=np.float32)
    transformed_array = np.asarray(transformed, dtype=np.float32)

    mse = float(np.mean((original_array - transformed_array) ** 2))
    normalized_mse = mse / (255.0 ** 2)

    return mse, normalized_mse


def prepare_image(image, max_size=512):
    """Resize an image to manageable dimensions divisible by 8."""
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))

    width = max(64, (image.width // 8) * 8)
    height = max(64, (image.height // 8) * 8)

    return image.resize((width, height), Image.Resampling.LANCZOS)


# --------------------------------------------------
# 3. Load pretrained Hugging Face model
# --------------------------------------------------

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True
    )

    pipe = pipe.to(device)

    # Reduce memory use when supported
    pipe.enable_attention_slicing()

    return pipe


# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------

st.title("🎨 AI Image Transformation Studio")

st.write(
    "Upload an image, choose transformation settings, "
    "and compare the original image with the AI-generated result."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["png", "jpg", "jpeg"]
)

prompt = st.text_input(
    "Transformation prompt",
    value=(
        "a beautiful watercolor painting, "
        "detailed, artistic brush strokes, vibrant colors"
    )
)

negative_prompt = st.text_input(
    "Negative prompt (optional)",
    value="blurry, low quality, distorted"
)

col_settings1, col_settings2 = st.columns(2)

with col_settings1:
    strength = st.slider(
        "Transformation strength",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=(
            "Lower values preserve more of the input. "
            "Higher values generally allow larger changes."
        )
    )

with col_settings2:
    steps = st.slider(
        "Inference steps",
        min_value=10,
        max_value=30,
        value=20,
        step=5
    )


# --------------------------------------------------
# 5. Show original image
# --------------------------------------------------

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")
        input_image = prepare_image(original)

        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

        if st.button("✨ Transform Image", type="primary"):

            if not prompt.strip():
                st.warning("Please enter a transformation prompt.")

            else:
                try:
                    with st.spinner(
                        "Loading model and transforming image. "
                        "The first run may take several minutes..."
                    ):
                        pipe = load_model()

                        # Fixed seed makes results reproducible
                        generator = torch.Generator(
                            device="cpu"
                        ).manual_seed(42)

                        result = pipe(
                            prompt=prompt,
                            negative_prompt=(
                                negative_prompt.strip() or None
                            ),
                            image=input_image,
                            strength=strength,
                            num_inference_steps=steps,
                            guidance_scale=7.0,
                            generator=generator
                        ).images[0]

                        # Compare both images at the same dimensions
                        original_compare = input_image.resize(
                            result.size,
                            Image.Resampling.LANCZOS
                        )

                        mse, normalized_mse = calculate_mse(
                            original_compare,
                            result
                        )

                    # Store results for display after reruns
                    st.session_state["original_result"] = (
                        original_compare
                    )
                    st.session_state["transformed_result"] = result
                    st.session_state["mse_result"] = mse
                    st.session_state["normalized_mse_result"] = (
                        normalized_mse
                    )

                except Exception as error:
                    st.error(f"Transformation failed: {error}")
                    st.info(
                        "Check disk space, internet connection, "
                        "model download, and available RAM."
                    )

    except Exception as error:
        st.error(f"Could not open the uploaded image: {error}")


# --------------------------------------------------
# 6. Display results and download
# --------------------------------------------------

if "transformed_result" in st.session_state:

    original_result = st.session_state["original_result"]
    transformed_result = st.session_state["transformed_result"]

    st.divider()
    st.subheader("Image Comparison")

    col_original, col_transformed = st.columns(2)

    with col_original:
        st.markdown("**Original Image**")
        st.image(
            original_result,
            use_container_width=True
        )

    with col_transformed:
        st.markdown("**Transformed Image**")
        st.image(
            transformed_result,
            use_container_width=True
        )

    st.subheader("Difference Metrics")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Mean Squared Error (MSE)",
            f"{st.session_state['mse_result']:.2f}"
        )

    with metric2:
        st.metric(
            "Normalized MSE",
            f"{st.session_state['normalized_mse_result']:.6f}"
        )

    st.caption(
        "MSE is the average squared difference between corresponding "
        "pixel values. Zero means the compared images are identical. "
        "Higher values indicate larger pixel-level differences."
    )

    st.download_button(
        label="⬇️ Download Transformed Image",
        data=image_to_bytes(transformed_result),
        file_name="transformed_image.png",
        mime="image/png"
    )
