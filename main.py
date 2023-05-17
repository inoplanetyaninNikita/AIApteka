from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showinfo

import pandas as pd
import seaborn as sns
from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


dataStart: DataFrame

root = Tk()
root.title("AI")
root.geometry("400x300")

root.grid_rowconfigure(index=0, weight=1)
root.grid_columnconfigure(index=0, weight=1)
root.grid_columnconfigure(index=1, weight=1)
root.grid_columnconfigure(index=2, weight=1)

# создаем список
columns_listbox = Listbox()
columns_listbox.grid(row=0, column=0, columnspan=2, sticky=EW, padx=5, pady=5)


# открываем файл в текстовое поле
def open_file():
    filepath = filedialog.askopenfilename()
    if filepath != "":
        with open(filepath, "r") as file:
            global dataStart
            columns_listbox.delete(0,'end')
            dataStart = pd.read_csv(file)
            for column in dataStart:
                columns_listbox.insert(END, column)
            columns_listbox.insert(END, "")

open_button = ttk.Button(text="Открыть файл", command=open_file)
open_button.grid(column=0, row=1, sticky=NSEW, padx=10)

# сохраняем текст из текстового поля в файл
def save_file():
    filepath = filedialog.asksaveasfilename()
    if filepath != "":
        text = text_editor.get("1.0", END)
        with open(filepath, "w") as file:
            file.write(text)

save_button = ttk.Button(text="Сохранить файл", command=save_file)
save_button.grid(column=1, row=1, sticky=NSEW, padx=10)

def test():
    columns_listbox.curselection()
    for i in columns_listbox.curselection():
        global dataStart
        answer_column_num = columns_listbox.get(i)
        dataStart = dataStart.dropna()

        def replace_with_number(word):
            return word_to_number.setdefault(word, len(word_to_number) + 1)

        # Проверяем, содержит ли столбец строки
        for column in dataStart:

            # Проверяем, есть ли string в столбце
            contains_string = dataStart[column].astype(str).str.contains('.*')
            if contains_string.any():

                # Проверяем, могут быть эти столбцы флоатом
                is_float = pd.to_numeric(dataStart[column], errors='coerce').notnull().all()
                is_datetime = pd.api.types.is_datetime64_any_dtype(dataStart[column])

                if is_float == False & is_datetime == False:
                    word_to_number = {}

                    # Заменяем строки на цифры
                    dataStart[column] = dataStart[column].apply(replace_with_number)

        answer_column = dataStart[answer_column_num]
        dataStart = dataStart.drop(answer_column_num, axis=1)

        rs_space = {'max_depth': list(np.arange(10, 100, step=10)) + [None],
                    'n_estimators': [10, 100, 200],
                    'max_features': randint(1, 7),
                    'min_samples_leaf': randint(1, 4),
                    'min_samples_split': np.arange(2, 10, step=2)
                    }

        X_train, X_test, y_train, y_test = train_test_split(dataStart, answer_column, test_size=0.8,
                                                            shuffle=True)
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(rf, rs_space, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3)
        model_random = rf_random.fit(X_train, y_train)

        # точность на тренировочной выборке
        print(model_random.score(X_train, y_train))

        # точность на тестовой выборке
        print(model_random.score(X_test, y_test))
        testScore = model_random.score(X_test, y_test)
        showinfo(title="AI", message="Точность тестовой выборки " + testScore)



learn_button = ttk.Button(text="Обучить модель", command=test)
learn_button.grid(column=1, row=2, sticky=NSEW, padx=10)


root.mainloop()




















