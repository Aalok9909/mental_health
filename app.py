# from flask import Flask,request,jsonify
# import numpy as np
# import pickle

# model = pickle.load(open('mind_ease_model.pkl','rb'))

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello world"

# @app.route('/predict',methods=['POST'])
# def predict():
#     Age = request.form.get('Age')
#     Gender = request.form.get('Gender')
#     Mood = request.form.get('Mood')
#     Activity_Type = request.form.get('Activity_Type')
#     Duration = request.form.get('Duration')
#     Exercise_Type = request.form.get('Exercise_Type')
#     Diet_Type = request.form.get('Diet_Type')
#     Goal = request.form.get('Goal')
#     Support_Type = request.form.get('Support_Type')
#     Sleep_Category = request.form.get('Sleep_Category')
#     # mood = request.form.get('Mood')


#     input_query = np.array([[Age,Gender,Mood,Activity_Type,Duration,Exercise_Type,Diet_Type,Goal,Support_Type,Sleep_Category]])
#     # result={'age':Age,'sleep_cateogry':Sleep_Category}
#     result = model.predict(input_query)[0]


#     return jsonify({'placement':str(result)})

# if __name__ == '__main__':
#     app.run(debug=True)






# from flask import Flask, request, jsonify
# import numpy as np
# import pickle

# # Load the trained model
# with open('mind_ease_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Ensure the model has a predict() method
# if not hasattr(model, 'predict'):
#     raise ValueError("Loaded model is not valid. It does not have a predict() method.")

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello, welcome to MindEase AI!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from request
#         Age = int(request.form.get('Age', 0))
#         Gender = int(request.form.get('Gender', 0))
#         Mood = int(request.form.get('Mood', 0))
#         Activity_Type = int(request.form.get('Activity_Type', 0))
#         Duration = int(request.form.get('Duration', 0))
#         Exercise_Type = int(request.form.get('Exercise_Type', 0))
#         Diet_Type = int(request.form.get('Diet_Type', 0))
#         Goal = int(request.form.get('Goal', 0))
#         Support_Type = int(request.form.get('Support_Type', 0))
#         Sleep_Category = int(request.form.get('Sleep_Category', 0))

#         # Convert to NumPy array and ensure correct type
#         input_query = np.array([[Age, Gender, Mood, Activity_Type, Duration,
#                                  Exercise_Type, Diet_Type, Goal, Support_Type, Sleep_Category]]).astype(np.float64)

#         # Predict the output
#         result = model.predict(input_query)[0]

#         return jsonify({'placement': str(result)})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# import numpy as np
# import pickle

# # Load trained model
# with open('mind_ease_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Ensure model has a predict() method
# if not hasattr(model, 'predict'):
#     raise ValueError("Loaded model is invalid. It does not have a predict() method.")

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Hello, welcome to MindEase AI!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Convert categorical inputs to numbers
#         mental_health_mapping = {'None': 0, 'Stress': 1, 'Anxiety': 2, 'Depression': 3}
#         therapist_mapping = {'No': 0, 'Yes': 1}
#         sleep_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

#         # Get form data and apply mappings
#         Age = int(request.form.get('Age', 0))
#         Gender = int(request.form.get('Gender', 0))
#         Mood = int(request.form.get('Mood', 0))
#         Activity_Type = int(request.form.get('Activity_Type', 0))
#         Duration = float(request.form.get('Duration', 0))
#         Exercise_Type = int(request.form.get('Exercise_Type', 0))
#         Diet_Type = int(request.form.get('Diet_Type', 0))
#         Goal = int(request.form.get('Goal', 0))
#         Support_Type = int(request.form.get('Support_Type', 0))
#         Mood_Before_Activity = int(request.form.get('Mood_Before_Activity', 0))
#         Mood_After_Activity = int(request.form.get('Mood_After_Activity', 0))

#         # Convert categorical inputs using mappings
#         Mental_Health_Condition = mental_health_mapping.get(request.form.get('Mental_Health_Condition', 'None'), 0)
#         Therapist_Interaction = therapist_mapping.get(request.form.get('Therapist_Interaction', 'No'), 0)
#         Sleep_Category = sleep_mapping.get(request.form.get('Sleep_Category', 'Moderate'), 1)

#         # Convert input into a NumPy array
#         input_query = np.array([[Age, Gender, Mood, Activity_Type, Duration, Exercise_Type, Diet_Type,
#                                  Goal, Support_Type, Mood_Before_Activity, Mood_After_Activity,
#                                  Mental_Health_Condition, Therapist_Interaction, Sleep_Category]]).astype(np.float64)

#         # Predict the output
#         result = model.predict(input_query)[0]

#         return jsonify({'prediction': str(result)})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the trained model
with open('mind_ease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ensure model has a predict() method
if not hasattr(model, 'predict'):
    raise ValueError("Error: Loaded model does not have a predict() method.")

app = Flask(__name__)

# Mappings for categorical values
mental_health_mapping = {'None': 0, 'Stress': 1, 'Anxiety': 2, 'Depression': 3}
therapist_mapping = {'No': 0, 'Yes': 1}
sleep_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}

@app.route('/')
def index():
    return "Welcome to MindEase AI Mental Health Prediction!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from request form
        Age = int(request.form.get('Age', 0))
        Gender = int(request.form.get('Gender', 0))
        Mood = int(request.form.get('Mood', 0))
        Activity_Type = int(request.form.get('Activity_Type', 0))
        Duration = float(request.form.get('Duration', 0))
        Exercise_Type = int(request.form.get('Exercise_Type', 0))
        Diet_Type = int(request.form.get('Diet_Type', 0))
        Goal = int(request.form.get('Goal', 0))
        Support_Type = int(request.form.get('Support_Type', 0))
        Mood_Before_Activity = int(request.form.get('Mood_Before_Activity', 0))
        Mood_After_Activity = int(request.form.get('Mood_After_Activity', 0))

        # Convert categorical inputs
        Therapist_Interaction = therapist_mapping.get(request.form.get('Therapist_Interaction', 'No'), 0)
        Sleep_Category = sleep_mapping.get(request.form.get('Sleep_Category', 'Moderate'), 1)

        # Create input array
        input_query = np.array([[Age, Gender, Mood, Activity_Type, Duration, Exercise_Type, Diet_Type,
                                 Goal, Support_Type, Mood_Before_Activity, Mood_After_Activity,
                                 Therapist_Interaction, Sleep_Category]]).astype(np.float64)

        # Model prediction
        prediction_index = model.Mental_Health_Condition(input_query)[0]
        predicted_label = [key for key, value in mental_health_mapping.items() if value == prediction_index][0]

        return jsonify({'Mental_Health_Condition': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
