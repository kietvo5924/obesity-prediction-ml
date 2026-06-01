import pickle
from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
import traceback # Import traceback để dùng trong exception

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'

# Load Model và Scaler
model_package_path = 'obesity_adaboost_full.pkl'
model_classifier = None
scaler = None
expected_features_order = None

try:
    with open(model_package_path, 'rb') as f:
        model_package = pickle.load(f)
    model_classifier = model_package['model']
    scaler = model_package['scaler']
    # Đã xóa print báo tải thành công

    expected_features_order = [
        'Sex', 'Age', 'Height', 'Overweight_Obese_Family', 'Consumption_of_Fast_Food',
        'Frequency_of_Consuming_Vegetables', 'Number_of_Main_Meals_Daily',
        'Food_Intake_Between_Meals', 'Smoking', 'Liquid_Intake_Daily',
        'Calculation_of_Calorie_Intake', 'Physical_Excercise',
        'Schedule_Dedicated_to_Technology', 'Type_of_Transportation_Used',
        'Activity_Level'
    ]
    # Đã xóa print thứ tự và số lượng đặc trưng

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{model_package_path}'.") # Giữ lại print lỗi
except Exception as e:
    print(f"Lỗi khi tải file model/scaler: {e}") # Giữ lại print lỗi

@app.route('/')
def home():
    return render_template('index.html', prediction=-1)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_classifier or not scaler or not expected_features_order:
        flash("Lỗi: Mô hình hoặc scaler chưa được tải.", "danger")
        return render_template('index.html', prediction=-2)

    if request.method == 'POST':
        try:
            form_data = {
                'Sex': int(request.form['Sex']),
                'Age': int(request.form['Age']),
                'Height': float(request.form['Height']),
                'Overweight_Obese_Family': int(request.form['Overweight_Obese_Family']),
                'Consumption_of_Fast_Food': int(request.form['Consumption_of_Fast_Food']),
                'Frequency_of_Consuming_Vegetables': int(request.form['Frequency_of_Consuming_Vegetables']),
                'Number_of_Main_Meals_Daily': int(request.form['Number_of_Main_Meals_Daily']),
                'Food_Intake_Between_Meals': int(request.form['Food_Intake_Between_Meals']),
                'Smoking': int(request.form['Smoking']),
                'Liquid_Intake_Daily': int(request.form['Liquid_Intake_Daily']),
                'Calculation_of_Calorie_Intake': int(request.form['Calculation_of_Calorie_Intake']),
                'Physical_Excercise': int(request.form['Physical_Excercise']),
                'Schedule_Dedicated_to_Technology': int(request.form['Schedule_Dedicated_to_Technology']),
                'Type_of_Transportation_Used': int(request.form['Type_of_Transportation_Used']),
            }

            form_data['Activity_Level'] = float(form_data['Physical_Excercise']) + float(form_data['Type_of_Transportation_Used'])

            input_df = pd.DataFrame([form_data])
            input_df = input_df[expected_features_order]
            # Đã xóa print input_df và số lượng cột

            data_scaled = scaler.transform(input_df)
            # Đã xóa print data_scaled

            predicted_class_numeric = model_classifier.predict(data_scaled)[0]
            # Đã xóa print kết quả dạng số

            prediction_map = {
                1: 'Underweight', 2: 'Normal', 3: 'Overweight', 4: 'Obesity'
            }
            predicted_class_label = prediction_map.get(predicted_class_numeric, "Unknown")
            # Đã xóa print kết quả dạng chữ

            flash("Dự đoán thành công!", "success")
            return render_template('index.html', prediction=predicted_class_label)

        except KeyError as e:
            print(f"Lỗi KeyError: Thiếu trường '{e}'") # Giữ lại print lỗi
            flash(f"Lỗi: Thiếu trường '{e}'.", "danger")
            return render_template('index.html', prediction=-2)
        except ValueError as e:
            print(f"Lỗi ValueError: Dữ liệu không hợp lệ. Lỗi: {e}") # Giữ lại print lỗi
            flash("Lỗi: Nhập giá trị số hợp lệ cho Chiều cao.", "danger")
            return render_template('index.html', prediction=-2)
        except Exception as e:
            print(f"Lỗi không xác định: {e}") # Giữ lại print lỗi
            traceback.print_exc() # Giữ lại traceback
            flash(f"Lỗi không mong muốn: {e}", "danger")
            return render_template('index.html', prediction=-2)

    return render_template('index.html', prediction=-1)

if __name__ == '__main__':
    app.run(debug=True, port=5001)