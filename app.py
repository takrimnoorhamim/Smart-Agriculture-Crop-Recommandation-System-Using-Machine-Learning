# ============================================================================
# SMART AGRICULTURE CROP RECOMMENDATION SYSTEM - MODERN UI
# ============================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

# ============================================================================
# Step 1: Load Saved Models and Encoders
# ============================================================================
print("\n" + "="*60)
print("LOADING MODELS AND ENCODERS...")
print("="*60)

try:
    model = joblib.load('best_crop_model.pkl')
    le_region = joblib.load('region_encoder.pkl')
    le_season = joblib.load('season_encoder.pkl')
    le_crop = joblib.load('crop_encoder.pkl')
    
    try:
        feature_columns = joblib.load('feature_columns.pkl')
        print("✓ Feature columns loaded successfully!")
    except FileNotFoundError:
        feature_columns = [
            'Rainfall_mm', 'Temperature_C', 'Humidity_pct', 
            'SoilMoisture_pct', 'Sunlight_hours', 'Soil_pH', 
            'SoilN_kg_ha', 'SoilP_kg_ha', 'SoilK_kg_ha',
            'Region_Encoded', 'Season_Encoded'
        ]
        print("✓ Using default feature order")
    
    print(f"\n✓ All models loaded successfully!")
    print(f"  Total crops: {len(le_crop.classes_)}")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Model files not found!")
    print(f"  Missing file: {e.filename}")
    input("\nPress Enter to exit...")
    exit()
except Exception as e:
    print(f"\n❌ ERROR loading models: {e}")
    input("\nPress Enter to exit...")
    exit()

# ============================================================================
# Step 2: Create Main Application Window
# ============================================================================
root = tk.Tk()
root.title("Smart Agriculture AI - Crop Recommendation")

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window to 90% of screen size for better visibility
window_width = int(screen_width * 0.9)
window_height = int(screen_height * 0.9)

root.geometry(f"{window_width}x{window_height}")
root.configure(bg="#0F172A")
root.resizable(True, True)  # Allow resizing

# Make it start maximized (optional - uncomment if you want)
# root.state('zoomed')  # For Windows
# root.attributes('-zoomed', True)  # For Linux

# ============================================================================
# Step 3: Modern Header with Gradient Effect
# ============================================================================
header_frame = tk.Frame(root, bg="#1E293B", height=100)
header_frame.pack(fill=tk.X)
header_frame.pack_propagate(False)

title_label = tk.Label(
    header_frame,
    text="🌾 SMART AGRICULTURE AI",
    font=("Segoe UI", 26, "bold"),
    bg="#1E293B",
    fg="#10B981"
)
title_label.pack(pady=15)

subtitle_label = tk.Label(
    header_frame,
    text="Intelligent Crop Recommendation System",
    font=("Segoe UI", 11),
    bg="#1E293B",
    fg="#94A3B8"
)
subtitle_label.pack()

# ============================================================================
# Step 4: Main Container with Two Panels
# ============================================================================
main_container = tk.Frame(root, bg="#0F172A")
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# LEFT PANEL - Input Controls (60% width)
left_panel = tk.Frame(main_container, bg="#1E293B")
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

# RIGHT PANEL - Live Prediction Results (40% width)
right_panel = tk.Frame(main_container, bg="#1E293B")
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

# ============================================================================
# Step 5: Scrollable Input Area with Custom Thin Scrollbar
# ============================================================================
# Custom style for thin scrollbar
style = ttk.Style()
style.theme_use('clam')
style.configure("Thin.Vertical.TScrollbar", 
                background="#10B981",
                troughcolor="#0F172A",
                bordercolor="#1E293B",
                arrowcolor="#E2E8F0",
                width=8)  # Thin scrollbar - 8 pixels wide
style.map("Thin.Vertical.TScrollbar",
          background=[('active', '#059669'), ('pressed', '#047857')])

canvas = tk.Canvas(left_panel, bg="#1E293B", highlightthickness=0)
scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview, 
                          style="Thin.Vertical.TScrollbar")
scrollable_frame = tk.Frame(canvas, bg="#1E293B")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
scrollbar.pack(side="right", fill="y", padx=(0, 5))

# ============================================================================
# Step 6: Input Variables with Smart Defaults
# ============================================================================
regions = le_region.classes_.tolist()
seasons = le_season.classes_.tolist()

input_vars = {}
slider_vars = {}
value_labels = {}

# Smart default values for quick demo
defaults = {
    'rainfall': 200.0,
    'temperature': 28.0,
    'humidity': 70.0,
    'soil_moisture': 45.0,
    'sunlight': 7.0,
    'soil_ph': 6.5,
    'nitrogen': 60.0,
    'phosphorus': 40.0,
    'potassium': 50.0
}

# Slider configurations
slider_configs = [
    # (key, label, min, max, default, unit, emoji)
    ('rainfall', 'Rainfall', 0, 361, defaults['rainfall'], 'mm', '🌧️'),
    ('temperature', 'Temperature', 17, 36, defaults['temperature'], '°C', '🌡️'),
    ('humidity', 'Humidity', 30, 100, defaults['humidity'], '%', '💧'),
    ('soil_moisture', 'Soil Moisture', 5, 78, defaults['soil_moisture'], '%', '💦'),
    ('sunlight', 'Sunlight', 2.5, 12, defaults['sunlight'], 'hrs', '☀️'),
    ('soil_ph', 'Soil pH', 4.5, 8.5, defaults['soil_ph'], '', '⚗️'),
    ('nitrogen', 'Nitrogen (N)', 5, 118, defaults['nitrogen'], 'kg/ha', '🧪'),
    ('phosphorus', 'Phosphorus (P)', 5, 70, defaults['phosphorus'], 'kg/ha', '🧪'),
    ('potassium', 'Potassium (K)', 5, 95, defaults['potassium'], 'kg/ha', '🧪')
]

row = 0

# ============================================================================
# Section 1: Location & Season
# ============================================================================
section1 = tk.Label(
    scrollable_frame,
    text="📍 LOCATION & SEASON",
    font=("Segoe UI", 13, "bold"),
    bg="#1E293B",
    fg="#10B981",
    anchor="w"
)
section1.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 15))
row += 1

# Region Dropdown
region_frame = tk.Frame(scrollable_frame, bg="#1E293B")
region_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

tk.Label(
    region_frame,
    text="🗺️ Region:",
    font=("Segoe UI", 11, "bold"),
    bg="#1E293B",
    fg="#E2E8F0"
).pack(side="left", padx=(0, 10))

input_vars['region'] = tk.StringVar(value=regions[0] if regions else "")
region_combo = ttk.Combobox(
    region_frame,
    textvariable=input_vars['region'],
    values=regions,
    state="readonly",
    width=30,
    font=("Segoe UI", 10)
)
region_combo.pack(side="left")
row += 1

# Season Dropdown
season_frame = tk.Frame(scrollable_frame, bg="#1E293B")
season_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

tk.Label(
    season_frame,
    text="🌦️ Season:",
    font=("Segoe UI", 11, "bold"),
    bg="#1E293B",
    fg="#E2E8F0"
).pack(side="left", padx=(0, 10))

input_vars['season'] = tk.StringVar(value=seasons[0] if seasons else "")
season_combo = ttk.Combobox(
    season_frame,
    textvariable=input_vars['season'],
    values=seasons,
    state="readonly",
    width=30,
    font=("Segoe UI", 10)
)
season_combo.pack(side="left")
row += 1

tk.Frame(scrollable_frame, bg="#334155", height=2).grid(
    row=row, column=0, columnspan=3, sticky="ew", pady=20
)
row += 1

# ============================================================================
# Section 2: Interactive Sliders
# ============================================================================
section2 = tk.Label(
    scrollable_frame,
    text="⚙️ ENVIRONMENTAL CONDITIONS",
    font=("Segoe UI", 13, "bold"),
    bg="#1E293B",
    fg="#10B981",
    anchor="w"
)
section2.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 15))
row += 1

def create_slider(parent_row, key, label, min_val, max_val, default, unit, emoji):
    """Create modern slider with live value display"""
    
    # Container
    container = tk.Frame(scrollable_frame, bg="#1E293B")
    container.grid(row=parent_row, column=0, columnspan=3, sticky="ew", pady=12)
    
    # Label and Value Display
    label_frame = tk.Frame(container, bg="#1E293B")
    label_frame.pack(fill=tk.X, pady=(0, 5))
    
    tk.Label(
        label_frame,
        text=f"{emoji} {label}",
        font=("Segoe UI", 10, "bold"),
        bg="#1E293B",
        fg="#E2E8F0",
        anchor="w"
    ).pack(side="left")
    
    # Value label
    value_label = tk.Label(
        label_frame,
        text=f"{default:.1f} {unit}",
        font=("Segoe UI", 10, "bold"),
        bg="#1E293B",
        fg="#10B981",
        anchor="e"
    )
    value_label.pack(side="right")
    value_labels[key] = value_label
    
    # Slider
    slider_var = tk.DoubleVar(value=default)
    slider_vars[key] = slider_var
    
    slider = ttk.Scale(
        container,
        from_=min_val,
        to=max_val,
        orient=tk.HORIZONTAL,
        variable=slider_var,
        command=lambda v: update_value(key, float(v), unit)
    )
    slider.pack(fill=tk.X, pady=(0, 5))
    
    # Range indicator
    tk.Label(
        container,
        text=f"Range: {min_val} - {max_val} {unit}",
        font=("Segoe UI", 8),
        bg="#1E293B",
        fg="#64748B"
    ).pack(anchor="w")
    
    return parent_row + 1

def update_value(key, value, unit):
    """Update value label and trigger prediction"""
    value_labels[key].config(text=f"{value:.1f} {unit}")
    # Auto-predict on change (with small delay to avoid too many calls)
    root.after(500, auto_predict)

# Create all sliders
for config in slider_configs:
    row = create_slider(row, *config)

# ============================================================================
# Step 7: RIGHT PANEL - Live Results Display (SCROLLABLE)
# ============================================================================
# Create scrollable canvas for right panel
right_canvas = tk.Canvas(right_panel, bg="#1E293B", highlightthickness=0)
right_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=right_canvas.yview,
                                style="Thin.Vertical.TScrollbar")
result_container = tk.Frame(right_canvas, bg="#1E293B")

result_container.bind(
    "<Configure>",
    lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
)

right_canvas.create_window((0, 0), window=result_container, anchor="nw", width=right_panel.winfo_width())

# Update canvas window width when right panel resizes
def update_canvas_width(event):
    right_canvas.itemconfig(right_canvas.find_all()[0], width=event.width - 30)

right_panel.bind("<Configure>", update_canvas_width)

right_canvas.configure(yscrollcommand=right_scrollbar.set)

# Enable mouse wheel for right panel
def _on_right_mousewheel(event):
    right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel)

right_canvas.pack(side="left", fill="both", expand=True, padx=(20, 5), pady=20)
right_scrollbar.pack(side="right", fill="y", padx=(0, 15), pady=20)

result_title = tk.Label(
    result_container,
    text="🎯 PREDICTION RESULTS",
    font=("Segoe UI", 14, "bold"),
    bg="#1E293B",
    fg="#10B981"
)
result_title.pack(pady=(0, 20))

# Recommended Crop Display
crop_frame = tk.Frame(result_container, bg="#0F172A", relief=tk.SOLID, bd=2)
crop_frame.pack(fill=tk.X, pady=10)

tk.Label(
    crop_frame,
    text="Recommended Crop",
    font=("Segoe UI", 10),
    bg="#0F172A",
    fg="#94A3B8"
).pack(pady=(10, 5))

recommended_crop_label = tk.Label(
    crop_frame,
    text="---",
    font=("Segoe UI", 24, "bold"),
    bg="#0F172A",
    fg="#10B981"
)
recommended_crop_label.pack(pady=(0, 10))

# Confidence Display
confidence_frame = tk.Frame(result_container, bg="#0F172A", relief=tk.SOLID, bd=2)
confidence_frame.pack(fill=tk.X, pady=10)

tk.Label(
    confidence_frame,
    text="Confidence Score",
    font=("Segoe UI", 10),
    bg="#0F172A",
    fg="#94A3B8"
).pack(pady=(10, 5))

confidence_label = tk.Label(
    confidence_frame,
    text="0.0%",
    font=("Segoe UI", 20, "bold"),
    bg="#0F172A",
    fg="#3B82F6"
)
confidence_label.pack(pady=(0, 10))

# Interpretation Message
interpretation_frame = tk.Frame(result_container, bg="#0F172A", relief=tk.SOLID, bd=2)
interpretation_frame.pack(fill=tk.X, pady=10)

tk.Label(
    interpretation_frame,
    text="💡 Why This Crop?",
    font=("Segoe UI", 11, "bold"),
    bg="#0F172A",
    fg="#94A3B8"
).pack(pady=(15, 10))

interpretation_text = tk.Text(
    interpretation_frame,
    font=("Segoe UI", 10),
    bg="#1E293B",
    fg="#E2E8F0",
    height=5,
    wrap=tk.WORD,
    relief=tk.FLAT,
    padx=15,
    pady=10
)
interpretation_text.pack(fill=tk.X, padx=10, pady=(0, 15))
interpretation_text.config(state=tk.DISABLED)

# Top 3 Alternative Crops
alternatives_frame = tk.Frame(result_container, bg="#0F172A", relief=tk.SOLID, bd=2)
alternatives_frame.pack(fill=tk.X, pady=10)

tk.Label(
    alternatives_frame,
    text="🌱 Alternative Crops",
    font=("Segoe UI", 11, "bold"),
    bg="#0F172A",
    fg="#94A3B8"
).pack(pady=(15, 10))

alternatives_text = tk.Text(
    alternatives_frame,
    font=("Segoe UI", 10),
    bg="#1E293B",
    fg="#E2E8F0",
    height=5,
    wrap=tk.WORD,
    relief=tk.FLAT,
    padx=15,
    pady=10
)
alternatives_text.pack(fill=tk.X, padx=10, pady=(0, 15))
alternatives_text.config(state=tk.DISABLED)

# Status Indicator
status_label = tk.Label(
    result_container,
    text="🟢 Ready for prediction",
    font=("Segoe UI", 9),
    bg="#1E293B",
    fg="#64748B"
)
status_label.pack(pady=(10, 0))

# ============================================================================
# Step 8: Prediction Functions
# ============================================================================

# Colorful Crop Emojis Dictionary
CROP_EMOJIS = {
    'Rice': '🌾',
    'Wheat': '🌾',
    'Maize': '🌽',
    'Corn': '🌽',
    'Soybean': '🌱',
    'Tomato': '🍅',
    'Chili': '🌶️',
    'Chilli': '🌶️',
    'Pepper': '🌶️',
    'Potato': '🥔',
    'Cotton': '🌱',
    'Sugarcane': '🎋',
    'Coffee': '☕',
    'Tea': '🍵',
    'Banana': '🍌',
    'Mango': '🥭',
    'Apple': '🍎',
    'Orange': '🍊',
    'Grapes': '🍇',
    'Watermelon': '🍉',
    'Coconut': '🥥',
    'Onion': '🧅',
    'Garlic': '🧄',
    'Carrot': '🥕',
    'Cabbage': '🥬',
    'Cauliflower': '🥦',
    'Pumpkin': '🎃',
    'Cucumber': '🥒',
    'Eggplant': '🍆',
    'Brinjal': '🍆',
    'Beans': '🫘',
    'Peas': '🫛',
    'Lentils': '🫘',
    'Chickpea': '🫘',
    'Sunflower': '🌻',
    'Mustard': '🌿',
    'Groundnut': '🥜',
    'Peanut': '🥜'
}

def get_crop_emoji(crop_name):
    """Get emoji for crop, with fallback"""
    # Check exact match
    if crop_name in CROP_EMOJIS:
        return CROP_EMOJIS[crop_name]
    
    # Check case-insensitive match
    for crop, emoji in CROP_EMOJIS.items():
        if crop.lower() in crop_name.lower() or crop_name.lower() in crop.lower():
            return emoji
    
    # Default emoji
    return '🌱'

def generate_interpretation(crop_name, rainfall, temp, humidity, moisture, sunlight, ph, n, p, k, season):
    """Generate intelligent interpretation message based on conditions"""
    
    reasons = []
    
    # Analyze rainfall
    if rainfall > 250:
        reasons.append(f"high rainfall ({rainfall:.0f}mm)")
    elif rainfall > 150:
        reasons.append(f"moderate rainfall ({rainfall:.0f}mm)")
    elif rainfall < 100:
        reasons.append(f"low rainfall ({rainfall:.0f}mm)")
    
    # Analyze temperature
    if temp > 32:
        reasons.append(f"warm temperature ({temp:.1f}°C)")
    elif temp < 22:
        reasons.append(f"cool temperature ({temp:.1f}°C)")
    elif 24 <= temp <= 30:
        reasons.append(f"optimal temperature ({temp:.1f}°C)")
    
    # Analyze humidity
    if humidity > 75:
        reasons.append(f"high humidity ({humidity:.0f}%)")
    elif humidity < 50:
        reasons.append(f"low humidity ({humidity:.0f}%)")
    
    # Analyze soil moisture
    if moisture > 60:
        reasons.append(f"high soil moisture ({moisture:.0f}%)")
    elif moisture < 30:
        reasons.append(f"low soil moisture ({moisture:.0f}%)")
    
    # Analyze sunlight
    if sunlight > 9:
        reasons.append(f"abundant sunlight ({sunlight:.1f} hrs)")
    elif sunlight < 5:
        reasons.append(f"limited sunlight ({sunlight:.1f} hrs)")
    
    # Analyze soil pH
    if 6.0 <= ph <= 7.0:
        reasons.append(f"neutral pH ({ph:.1f})")
    elif ph < 6.0:
        reasons.append(f"acidic soil (pH {ph:.1f})")
    elif ph > 7.5:
        reasons.append(f"alkaline soil (pH {ph:.1f})")
    
    # Analyze NPK levels
    if n > 80:
        reasons.append("rich nitrogen")
    if p > 50:
        reasons.append("good phosphorus")
    if k > 70:
        reasons.append("high potassium")
    
    # Add season
    reasons.append(f"{season} season")
    
    # Build message
    if len(reasons) >= 3:
        main_reasons = ", ".join(reasons[:3])
        message = f"{crop_name} is suitable due to {main_reasons}."
    elif len(reasons) > 0:
        main_reasons = " and ".join(reasons)
        message = f"{crop_name} is recommended because of {main_reasons}."
    else:
        message = f"{crop_name} matches your current environmental conditions."
    
    return message

def auto_predict():
    """Auto-predict when sliders change"""
    try:
        status_label.config(text="🔄 Predicting...", fg="#F59E0B")
        root.update()
        
        # Get values
        region_encoded = le_region.transform([input_vars['region'].get()])[0]
        season_encoded = le_season.transform([input_vars['season'].get()])[0]
        
        # Collect slider values
        rainfall = slider_vars['rainfall'].get()
        temperature = slider_vars['temperature'].get()
        humidity = slider_vars['humidity'].get()
        soil_moisture = slider_vars['soil_moisture'].get()
        sunlight = slider_vars['sunlight'].get()
        soil_ph = slider_vars['soil_ph'].get()
        nitrogen = slider_vars['nitrogen'].get()
        phosphorus = slider_vars['phosphorus'].get()
        potassium = slider_vars['potassium'].get()
        
        values = [rainfall, temperature, humidity, soil_moisture,
                 sunlight, soil_ph, nitrogen, phosphorus, potassium,
                 region_encoded, season_encoded]
        
        # Create features
        features = np.array([values])
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # Predict
        prediction = model.predict(features_df)[0]
        predicted_crop = le_crop.inverse_transform([prediction])[0]
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(features_df)[0]
            confidence = float(probabilities[prediction]) * 100
            
            # Top 3
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_crops = [(le_crop.inverse_transform([idx])[0], 
                           probabilities[idx] * 100) 
                          for idx in top_3_indices]
        except:
            confidence = 0.0
            top_3_crops = [(predicted_crop, 0.0)]
        
        # Update display with COLORFUL EMOJI
        crop_emoji = get_crop_emoji(predicted_crop)
        recommended_crop_label.config(text=f"{crop_emoji} {predicted_crop.upper()}")
        confidence_label.config(text=f"{confidence:.1f}%")
        
        # Color based on confidence
        if confidence >= 80:
            conf_color = "#10B981"
        elif confidence >= 60:
            conf_color = "#3B82F6"
        elif confidence >= 40:
            conf_color = "#F59E0B"
        else:
            conf_color = "#EF4444"
        confidence_label.config(fg=conf_color)
        
        # Generate and display interpretation message
        interpretation = generate_interpretation(
            predicted_crop, rainfall, temperature, humidity, 
            soil_moisture, sunlight, soil_ph, 
            nitrogen, phosphorus, potassium, 
            input_vars['season'].get()
        )
        interpretation_text.config(state=tk.NORMAL)
        interpretation_text.delete("1.0", tk.END)
        interpretation_text.insert("1.0", interpretation)
        interpretation_text.config(state=tk.DISABLED)
        
        # Update alternatives with EMOJIS
        alternatives_text.config(state=tk.NORMAL)
        alternatives_text.delete("1.0", tk.END)
        
        # Add each alternative crop with emoji and formatting
        for i, (crop, prob) in enumerate(top_3_crops, 1):
            alt_emoji = get_crop_emoji(crop)
            line = f"{i}. {alt_emoji} {crop.title()}: {prob:.1f}%\n"
            alternatives_text.insert(tk.END, line)
        
        alternatives_text.config(state=tk.DISABLED)
        
        status_label.config(text="🟢 Prediction complete", fg="#10B981")
        
        # Debug: Print to console
        print(f"✓ Predicted: {predicted_crop} ({confidence:.1f}%)")
        print(f"✓ Alternatives: {[c[0] for c in top_3_crops]}")
        
    except Exception as e:
        status_label.config(text="🔴 Prediction error", fg="#EF4444")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def manual_predict():
    """Manual prediction with detailed results"""
    auto_predict()
    crop_name = recommended_crop_label.cget('text')
    messagebox.showinfo(
        "Prediction Complete",
        f"✓ Crop recommendation updated!\n\n"
        f"Recommended: {crop_name}\n"
        f"Confidence: {confidence_label.cget('text')}"
    )

def reset_all():
    """Reset to default values"""
    for key, default in defaults.items():
        slider_vars[key].set(default)
    
    input_vars['region'].set(regions[0] if regions else "")
    input_vars['season'].set(seasons[0] if seasons else "")
    
    auto_predict()
    messagebox.showinfo("Reset", "✓ All values reset to defaults!")

# ============================================================================
# Step 9: Action Buttons
# ============================================================================
button_frame = tk.Frame(left_panel, bg="#1E293B")
button_frame.pack(side=tk.BOTTOM, pady=20)

predict_btn = tk.Button(
    button_frame,
    text="🔍 PREDICT NOW",
    command=manual_predict,
    font=("Segoe UI", 12, "bold"),
    bg="#10B981",
    fg="white",
    width=15,
    height=2,
    relief=tk.FLAT,
    cursor="hand2",
    activebackground="#059669"
)
predict_btn.grid(row=0, column=0, padx=8)

reset_btn = tk.Button(
    button_frame,
    text="↻ RESET",
    command=reset_all,
    font=("Segoe UI", 12, "bold"),
    bg="#6366F1",
    fg="white",
    width=15,
    height=2,
    relief=tk.FLAT,
    cursor="hand2",
    activebackground="#4F46E5"
)
reset_btn.grid(row=0, column=1, padx=8)

# ============================================================================
# Step 10: Initialize with Default Prediction
# ============================================================================
print("\n" + "="*60)
print("🚀 MODERN GUI APPLICATION STARTED!")
print("="*60)
print(f"✓ Loaded {len(le_crop.classes_)} crops")
print(f"✓ Interactive sliders enabled")
print(f"✓ Auto-prediction enabled")
print(f"✓ Window: {window_width}x{window_height}")
print("="*60 + "\n")

# Window is already centered and sized, no need to reposition
# Trigger initial prediction
root.after(1000, auto_predict)

# Bind dropdown changes to auto-predict
region_combo.bind('<<ComboboxSelected>>', lambda e: auto_predict())
season_combo.bind('<<ComboboxSelected>>', lambda e: auto_predict())

root.mainloop()