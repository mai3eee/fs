#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-python-headless')


# In[10]:


# Complete Fashion Recommendation System (with Skin Tone Detection)

import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------
# STEP 1: Load the Dataset and Clean It
# -------------------------------------------

# Load your dataset
dataset_path = 'fashion_dataset_dirty_improved.csv'  # Adjust path
data = pd.read_csv(dataset_path)

# Clean the dataset
critical_columns = ['gender', 'masterCategory', 'baseColour', 'usage']
data_clean = data.dropna(subset=critical_columns)

imputer = SimpleImputer(strategy='most_frequent')
data_clean[['season', 'subCategory', 'articleType']] = imputer.fit_transform(data_clean[['season', 'subCategory', 'articleType']])
data_clean['year'] = data_clean['year'].fillna(method='ffill')
data_clean.columns = [col.lower() for col in data_clean.columns]


# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Upload image
image_path = 'your_photo.jpg'  # <-- Replace with your actual face image file
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check your path!")

# Detect face
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    raise ValueError("No face detected. Try another photo!")

# Crop forehead
for (x, y, w, h) in faces:
    forehead = img[y:y + int(h * 0.25), x + int(w * 0.3):x + int(w * 0.7)]
    break  # Only first face

# Show forehead
plt.imshow(cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB))
plt.title("Detected Forehead Region")
plt.axis('off')
plt.show()

# Average RGB
forehead_rgb = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
average_rgb = forehead_rgb.mean(axis=(0,1))

# Define Reference Skin Tones
reference_skin_tones = {
    'Deep': np.array([66, 37, 17]),
    'Tan': np.array([157, 122, 84]),
    'Fair': np.array([229, 200, 166]),
    'Pale': np.array([243, 240, 240])
}

# Match closest tone
def closest_skin_tone(avg_rgb, ref_tones):
    min_dist = float('inf')
    closest = None
    for tone, ref_rgb in ref_tones.items():
        dist = np.linalg.norm(avg_rgb - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest = tone
    return closest

detected_skin_tone = closest_skin_tone(average_rgb, reference_skin_tones)
print(f"\n✅ Detected Skin Tone: {detected_skin_tone}")

# -------------------------------------------
# STEP 3: Outfit Recommendation based on Detected Skin Tone
# -------------------------------------------

# Define Skin Tone → Color Palette Mapping
def get_recommended_colors(skin_tone):
    skin_tone_to_color_palette = {
        'Deep': ['Navy Blue', 'Olive', 'Rust', 'Burgundy', 'Gold'],
        'Tan': ['Coral', 'Sea Green', 'Turquoise Blue', 'Yellow', 'Peach'],
        'Fair': ['Lavender', 'Soft Pink', 'Light Blue', 'Mint Green', 'White'],
        'Pale': ['Off White', 'Silver', 'Pastel Blue', 'Light Grey']
    }
    return skin_tone_to_color_palette.get(skin_tone, [])

# Recommendation Function
def recommend_outfits(gender, occasion, skin_tone, dataset):
    recommended_colors = get_recommended_colors(skin_tone)
    
    occasion_mapping = {
        'Casual': ['Casual', 'Travel', 'Sports'],
        'Formal': ['Formal'],
        'Party': ['Party'],
        'Ethnic': ['Ethnic'],
        'Sports': ['Sports'],
        'Travel': ['Travel']
    }
    allowed_usages = occasion_mapping.get(occasion, ['Casual'])
    
    filtered = dataset[
        (dataset['gender'].str.lower() == gender.lower()) &
        (dataset['basecolour'].isin(recommended_colors)) &
        (dataset['usage'].isin(allowed_usages))
    ]
    
    topwear = filtered[filtered['subcategory'].str.contains('topwear', case=False, na=False)]
    bottomwear = filtered[filtered['subcategory'].str.contains('bottomwear', case=False, na=False)]
    footwear = filtered[filtered['mastercategory'].str.contains('footwear', case=False, na=False)]
    
    outfit_combinations = []
    
    if not topwear.empty and not bottomwear.empty and not footwear.empty:
        for top in topwear.sample(min(5, len(topwear))).itertuples():
            for bottom in bottomwear.sample(min(3, len(bottomwear))).itertuples():
                for foot in footwear.sample(min(2, len(footwear))).itertuples():
                    outfit_combinations.append({
                        'Topwear': top.productdisplayname,
                        'Bottomwear': bottom.productdisplayname,
                        'Footwear': foot.productdisplayname
                    })
    return outfit_combinations

# -------------------------------------------
# STEP 4: Collect User Gender and Occasion, Then Recommend
# -------------------------------------------

# Manually ask gender and occasion
print("\n🎯 Now Let's Generate Outfit Suggestions!")
gender = input("Enter your Gender (Men/Women): ").strip()
occasion = input("Enter Occasion (Casual/Formal/Party/Ethnic/Sports/Travel): ").strip()

outfits = recommend_outfits(gender, occasion, detected_skin_tone, data_clean)

# -------------------------------------------
# STEP 5: Display Results
# -------------------------------------------

if outfits:
    print(f"\n🎉 Outfit Recommendations for {gender} - {occasion} Look (Detected Skin Tone: {detected_skin_tone})\n")
    for idx, outfit in enumerate(outfits[:5]):
        print(f"Outfit {idx + 1}:")
        print(f" - Topwear: {outfit['Topwear']}")
        print(f" - Bottomwear: {outfit['Bottomwear']}")
        print(f" - Footwear: {outfit['Footwear']}")
        print("-" * 50)
else:
    print("\n❌ No matching outfits found. Try selecting a different occasion!")


# In[ ]:




