import csv
import cv2

def search_number_in_csv(number, filename):
    results = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header row
        next(reader)
        for row in reader:
            if row[0] == str(number):
                results.append(row[1])
    return results

user_input = input("Digite o seu número de corrida: ")
# Search for the number in the CSV file
filename = 'number-images.csv'
texts = search_number_in_csv(user_input, filename)
if texts:
    for text in texts:
        image = cv2.imread(f'./photos/{text}')
        cv2.imshow(text, image)
        cv2.setWindowProperty(text, cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
     print(f"Não foram encontradas imagens para o número de corrida {user_input}.")