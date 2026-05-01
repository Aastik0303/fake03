import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# TensorFlow Lite aur Image processing imports
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# LangChain aur LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
import cv2

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================
# TFLITE MODEL LOADING
# ============================
MODEL_PATH = 'models/deepfake_detector_model.tflite'
interpreter = None
input_details = None
output_details = None

def load_cnn_model():
    """Load the pre-trained deepfake detection TFLite model"""
    global interpreter, input_details, output_details
    try:
        if os.path.exists(MODEL_PATH):
            # TFLite interpreter setup
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(f"✓ TFLite Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠ Warning: Model file not found at {MODEL_PATH}")
            interpreter = None
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        interpreter = None

# Load model at startup
load_cnn_model()

# ============================
# IMAGE PREPROCESSING
# ============================
def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess image for TFLite model"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

# ============================
# TFLITE PREDICTION LOGIC
# ============================
def predict_deepfake(img_path):
    """Run TFLite prediction on image"""
    try:
        processed_img = preprocess_image(img_path)
        
        if interpreter is not None:
            interpreter.set_tensor(input_details[0]['index'], processed_img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = float(prediction[0][0])
            # Logic: confidence > 0.7 is Fake
            is_fake = confidence > 0.7
            confidence_percent = confidence * 100 if is_fake else (1 - confidence) * 100
            
            return {
                'is_fake': is_fake,
                'confidence': confidence_percent,
                'raw_score': confidence
            }
        else:
            # Fallback mock logic
            return {'is_fake': False, 'confidence': 0.0, 'raw_score': 0.0}
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

# ============================
# LANGCHAIN TOOL & AGENT
# ============================
def deepfake_detection_tool_func(image_path: str) -> str:
    """Tool function for LangChain agent"""
    try:
        result = predict_deepfake(image_path)
        classification = "Deepfake" if result['is_fake'] else "Real"
        return f"Result: {classification} ({result['confidence']:.1f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"

deepfake_tool = Tool(
    name="DeepfakeDetectionTool",
    func=deepfake_detection_tool_func,
    description="Analyze an image for deepfake detection. Input: file path."
)

def create_deepfake_agent():
    """Initialize LangGraph agent with Groq"""
    llm = ChatGroq(
        model="llama-3.1-70b-versatile", 
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return create_react_agent(llm, tools=[deepfake_tool])

agent_executor = None
agent_error_message = "No error"
try:
    agent_executor = create_deepfake_agent()
    print("✓ Agent initialized")
except Exception as e:
    agent_error_message = str(e)

# ============================
# FLASK ROUTES
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Main analysis endpoint with automatic cleanup"""
    filepath = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if agent_executor is None:
            return jsonify({'error': f'Agent error: {agent_error_message}'}), 500
        
        # System instructions pass karein
        system_instructions = SystemMessage(content="You are an expert AI assistant specialized in deepfake detection. Explain the technical analysis results clearly.")
        
        response = agent_executor.invoke({
            "messages": [system_instructions, HumanMessage(content=f"Analyze image: {filepath}")]
        })
        
        return jsonify({'success': True, 'analysis': response["messages"][-1].content})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup ensures no storage leak
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': interpreter is not None,
        'agent_initialized': agent_executor is not None,
        'exact_error_reason': agent_error_message
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
