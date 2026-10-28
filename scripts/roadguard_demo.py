from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "roadguard_experiment_2"
    / "weights"
    / "best.pt"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="RoadGuard",
    page_icon="🚗",
    layout="wide"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🚗 RoadGuard")
st.subheader("Road Scene Object Detection")

st.write(
    "Upload a road image and RoadGuard will detect vehicles, "
    "people, traffic lights, and traffic signs."
)


# ---------------------------------------------------------
# CHECK MODEL
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        f"Model not found:\n{MODEL_PATH}"
    )
    st.stop()


model = load_model()


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# DETECTION
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("### Original Image")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Detect Objects"):

        with st.spinner("Running RoadGuard..."):

            results = model.predict(
                source=image,
                imgsz=416,
                conf=0.25,
                device="cpu"
            )

        result = results[0]

        # Annotated image
        annotated_image = result.plot()

        # Convert BGR → RGB for Streamlit
        annotated_image = annotated_image[:, :, ::-1]

        st.write("### RoadGuard Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )


        # -------------------------------------------------
        # DETECTION RESULTS
        # -------------------------------------------------

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:

            st.warning("No objects detected.")

        else:

            st.write("### Detected Objects")

            detections = []

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append(
                    {
                        "Object": class_name,
                        "Confidence": f"{confidence:.1%}"
                    }
                )

            st.table(detections)

            st.success(
                f"{len(detections)} object(s) detected."
            )