import pickle


with open('recommend.pkl', 'rb') as file:
    model = pickle.load(file)


N = int(input("Enter Nitrogen (N) value: "))
P = int(input("Enter Phosphorous (P) value: "))
K = int(input("Enter Potassium (K) value: "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))


inputs = [[N, P, K, temperature, humidity, ph, rainfall]]


predict = model.predict(inputs)


crop_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya', 7: 'Orange',
    8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes', 12: 'Mango', 13: 'Banana',
    14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans',
    19: 'Pigeonpeas', 20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'
}

if predict[0] in crop_dict:
    crop = crop_dict[predict[0]]
    print(f"{crop} is the best crop to be cultivated.")
else:
    print("Sorry, we are not able to recommend a proper crop for this environment.")
