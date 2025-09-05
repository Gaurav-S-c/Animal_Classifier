import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Class Names
with open("class_names.txt") as f:
    content = f.read().strip()
    if "," in content:
        class_names = content.split(",")
    else:
        class_names = content.splitlines()

num_classes = len(class_names)

# Loading pre_trainded model
class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
    self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
    self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)

    self.pooling=nn.MaxPool2d(2,2)

    self.relu=nn.ReLU()

    self.flatten=nn.Flatten()
    self.linear=nn.Linear((128*16*16),128)

    self.output=nn.Linear(128,num_classes)

  def forward(self,x):
    x=self.conv1(x) # ->(32,128,128)
    x=self.pooling(x) #-> (32,64,64)
    x=self.relu(x)

    x=self.conv2(x) # ->(64,64,64)
    x=self.pooling(x) #->(64,32,32)
    x=self.relu(x)

    x=self.conv3(x) #->(128,32,32)
    x=self.pooling(x) #->(128,16,16)
    x=self.relu(x)

    x=self.flatten(x)
    x=self.linear(x)
    x=self.output(x)
    return x
  
# Loading Trained Weights
model = Net()
model.load_state_dict(torch.load("model_state.pth", map_location=torch.device("cpu")))
model.eval()
  
# Define Transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),  
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🐾 Animal Classifier App")
st.markdown("Upload an animal image to detect whether it’s a **Cat 🐱, Dog 🐶, or Wild 🦁**")

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Add predict button
    if st.button("🔍 Predict"):
        # preprocess
        img_tensor = transform(image).unsqueeze(0)

        # predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]

        st.success(f"✨ Prediction: **{label}**")

