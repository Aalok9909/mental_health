
# # from flask import Flask, request, jsonify
# # import numpy as np
# # import pickle

# # # Load the trained model
# # with open('mind_ease_model.pkl', 'rb') as f:
# #     model = pickle.load(f)

# # # # Ensure model has a predict() method
# # # if not hasattr(model, 'predict'):
# # #     raise ValueError("Error: Loaded model does not have a predict() method.")

# # app = Flask(__name__)

# # # Mappings for categorical values
# # mental_health_mapping = {'None': 0, 'Stress': 1, 'Anxiety': 2, 'Depression': 3}
# # therapist_mapping = {'No': 0, 'Yes': 1}
# # sleep_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

# # @app.route('/')
# # def index():
# #     return "Welcome to MindEase AI Mental Health Prediction!"

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Get input values from request form
# #         Age = int(request.form.get('Age', 0))
# #         Gender = int(request.form.get('Gender', 0))
# #         Mood = int(request.form.get('Mood', 0))
# #         Activity_Type = int(request.form.get('Activity_Type', 0))
# #         Duration = float(request.form.get('Duration', 0))
# #         Exercise_Type = int(request.form.get('Exercise_Type', 0))
# #         Diet_Type = int(request.form.get('Diet_Type', 0))
# #         Goal = int(request.form.get('Goal', 0))
# #         Support_Type = int(request.form.get('Support_Type', 0))
# #         Mood_Before_Activity = int(request.form.get('Mood_Before_Activity', 0))
# #         Mood_After_Activity = int(request.form.get('Mood_After_Activity', 0))

# #         # Convert categorical inputs
# #         Therapist_Interaction = therapist_mapping.get(request.form.get('Therapist_Interaction', 'No'), 0)
# #         Sleep_Category = sleep_mapping.get(request.form.get('Sleep_Category', 'Moderate'), 1)

# #         # Create input array
# #         input_query = np.array([[Age, Gender, Mood, Activity_Type, Duration, Exercise_Type, Diet_Type,
# #                                  Goal, Support_Type, Mood_Before_Activity, Mood_After_Activity,
# #                                  Therapist_Interaction, Sleep_Category]]).astype(np.float64)

# #         # Model prediction
# #         prediction_index = model.Mental_Health_Condition(input_query)[0]
# #         predicted_label = [key for key, value in mental_health_mapping.items() if value == prediction_index][0]

# #         return jsonify({'Mental_Health_Condition': predicted_label})

# #     except Exception as e:
# #         return jsonify({'error': str(e)})

# # if __name__ == '__main__':
# #     app.run(debug=True)














# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import pickle
# import os

# # 🔹 Define the model file path (CHANGE THIS to your actual model path)
# MODEL_PATH = "C:/Users/SHAH KANHAIYALAL/Desktop/full_setup/mental_health_model.pkl"

# # 🔹 Check if the model file exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# # 🔹 Load the trained model
# with open(MODEL_PATH, "rb") as model_file:
#     model = pickle.load(model_file)

# # 🔹 Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for cross-origin requests

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Mental Health Prediction API is running!"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # 🔹 Get JSON data from request
#         data = request.get_json()
        
#         # 🔹 Required input features
#         required_features = [
#             "Age", "Gender", "Mood", "Activity_Type", "Duration",
#             "Exercise_Type", "Diet_Type", "Goal", "Support_Type",
#             "Mood_Before_Activity", "Mood_After_Activity", "Therapist_Interaction", "Sleep_Category"
#         ]
        
#         # 🔹 Convert JSON data to DataFrame
#         input_data = pd.DataFrame([data])

#         # 🔹 Ensure correct column ordering
#         input_data = input_data[required_features]

#         # 🔹 Make prediction
#         prediction = model.predict(input_data)[0]

#         # 🔹 Return response
#         return jsonify({"Mental_Health_Condition": str(prediction)})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)










#main code like numbers


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import pickle
# import os

# # 🔹 Load the trained model and class mapping
# MODEL_PATH = "C:/Users/SHAH KANHAIYALAL/Desktop/full_setup/mental_health_model.pkl"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# with open(MODEL_PATH, "rb") as model_file:
#     model_data = pickle.load(model_file)
#     model = model_data["model"]
#     class_mapping = model_data["mapping"]  # Load label mapping

# app = Flask(__name__)
# CORS(app)

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Mental Health Prediction API is running!"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         required_features = [
#             "Age", "Gender", "Mood", "Activity_Type", "Duration",
#             "Exercise_Type", "Diet_Type", "Goal", "Support_Type",
#             "Mood_Before_Activity", "Mood_After_Activity", "Therapist_Interaction", "Sleep_Category"
#         ]
        
#         # 🔹 Convert input into DataFrame
#         input_data = pd.DataFrame([data])
#         input_data = input_data[required_features]

#         # 🔹 Make prediction
#         prediction_numeric = model.predict(input_data)[0]

#         # 🔹 Convert number to original label
#         prediction_label = class_mapping[prediction_numeric]

#         return jsonify({"Mental_Health_Condition": prediction_label})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import pickle
# import os

# # 🔹 Load the trained model and class mapping
# MODEL_PATH = "mentals.pkl"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# with open(MODEL_PATH, "rb") as model_file:
#     model_data = pickle.load(model_file)
#     model = model_data["model"]
#     class_mapping = model_data["mapping"]  # Load label mapping

# app = Flask(__name__)
# CORS(app)

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Mental Health Prediction API is running!"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         required_features = [
#             "Age", "Gender", "Mood", "Activity_Type", "Duration",
#             "Exercise_Type", "Diet_Type", "Goal", "Support_Type",
#             "Mood_Before_Activity", "Mood_After_Activity", "Therapist_Interaction", "Sleep_Category"
#         ]
        
#         # 🔹 Convert input into DataFrame
#         input_data = pd.DataFrame([data])
#         input_data = input_data[required_features]

#         # 🔹 Make prediction
#         prediction_numeric = model.predict(input_data)[0]

#         # 🔹 Convert number to original label (Fixing the issue here)
#         prediction_label = class_mapping.get(prediction_numeric, "Unknown Condition")

#         return jsonify({
#             "Mental_Health_Condition": prediction_label,
#             "Predicted_Numeric_Value": int(prediction_numeric)  # Also return numeric for debugging
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# if __name__ == "__main__":
#     app.run(port=5000, debug=True)





import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# 🔹 Load Model & Label Encoder
with open("mental_health_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("mental_health_labels.pkl", "rb") as label_file:
    y_labels = pickle.load(label_file)

# 🔹 Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

@app.route("/")
def home():
    return jsonify({"message": "Mental Health Prediction API is running!"})

# 🔹 API Route for Predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON Data from Postman
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure model has a `predict()` method
        if not hasattr(model, "predict"):
            return jsonify({"error": "Loaded model is invalid. It does not have a predict() method."}), 400

        # Make Prediction
        prediction = model.predict(df)

        # Convert Numeric Prediction Back to Label
        predicted_label = y_labels[prediction[0]]

        return jsonify({"Mental_Health_Condition": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
