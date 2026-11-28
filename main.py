
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================

df = pd.read_csv("student_data.csv")

# Определяем категорию по экзаменационному баллу
def get_category(score):
    if score <= 65:
        return "A"
    elif score <= 80:
        return "B"
    return "C"


df["category"] = df["exam_score"].apply(get_category)

print("Размер таблицы:", df.shape)
print("\nПервые 5 студентов:")
print(df.head())

print("\nКоличество студентов по категориям:")
print(df["category"].value_counts())

print("\nПропущенные значения:")
print(df.isnull().sum())

print("\nСтатистика экзаменационных баллов:")
print(df["exam_score"].describe())


# ==========================================
# 2. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ (EDA)
# ==========================================

# Анализ числового фактора
def analyze_numeric(column, title, xlabel):
    correlation = df[column].corr(df["exam_score"])

    print(f"\n{title}")
    print(f"Корреляция с оценкой: {correlation:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Зависимость оценки от фактора
    sns.regplot(
        data=df, x=column, y="exam_score",
        ax=axes[0],
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"}
    )

    axes[0].set_title(f"{title} и оценка")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Балл за экзамен")

    # Сравнение категорий A, B, C
    sns.boxplot(
        data=df, x="category", y=column,
        order=["A", "B", "C"], ax=axes[1]
    )

    axes[1].set_title(f"{title} по категориям")
    axes[1].set_xlabel("Категория")
    axes[1].set_ylabel(xlabel)

    plt.tight_layout()
    plt.show()


# Анализ категориального фактора
def analyze_category(column, title, order):
    print(f"\n{title}: средние баллы")
    print(df.groupby(column)["exam_score"].mean().round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Распределение экзаменационных баллов
    sns.boxplot(
        data=df, x=column, y="exam_score",
        order=order, ax=axes[0]
    )

    axes[0].set_title(f"{title} и оценка")
    axes[0].set_xlabel(title)
    axes[0].set_ylabel("Балл за экзамен")

    # Процент студентов в каждой категории
    categories = pd.crosstab(
        df[column], df["category"],
        normalize="index"
    ) * 100

    categories = categories.reindex(
        index=order, columns=["A", "B", "C"]
    )

    categories.plot(
        kind="bar", stacked=True, ax=axes[1]
    )

    axes[1].set_title("Распределение категорий")
    axes[1].set_xlabel(title)
    axes[1].set_ylabel("Процент студентов")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Категория")

    plt.tight_layout()
    plt.show()


# 2.1. Время учёбы
analyze_numeric(
    "study_hours_per_day",
    "Время учёбы",
    "Часов учёбы в день"
)

# 2.2. Социальные сети
analyze_numeric(
    "social_media_hours",
    "Социальные сети",
    "Часов в социальных сетях"
)

# 2.3. Физическая активность
analyze_numeric(
    "exercise_frequency",
    "Физическая активность",
    "Частота физических упражнений"
)

# 2.4. Подработка
analyze_category(
    "part_time_job",
    "Подработка",
    ["No", "Yes"]
)

# 2.5. Пол студента
analyze_category(
    "gender",
    "Пол студента",
    ["Female", "Male"]
)


# ==========================================
# 3. РЕГРЕССИОННЫЙ АНАЛИЗ
# ==========================================

# Признаки и экзаменационные баллы
X = df.drop(columns=["student_id", "exam_score", "category"])
y = df["exam_score"]

# Разделяем числовые и текстовые признаки
numeric_cols = X.select_dtypes(include="number").columns
category_cols = X.select_dtypes(exclude="number").columns

# Обработка текстовых признаков
category_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Подготовка данных
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", category_transformer, category_cols)
])

# Создаём модель линейной регрессии
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучаем модель
model.fit(X_train, y_train)

# Проверяем качество прогнозирования
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nРЕЗУЛЬТАТЫ РЕГРЕССИИ")
print(f"Средняя ошибка (MAE): {mae:.2f}")
print(f"Коэффициент R²: {r2:.3f}")


# ==========================================
# 4. ПРОГНОЗ ДЛЯ НОВОГО СТУДЕНТА
# ==========================================

new_student = pd.DataFrame([{
    "age": 17,
    "gender": "Female",
    "study_hours_per_day": 2.9,
    "social_media_hours": 1.0,
    "online_movie_view_hours": 2.4,
    "part_time_job": "Yes",
    "attendance_percentage": 86.0,
    "sleep_hours": 6.8,
    "diet_quality": "Poor",
    "exercise_frequency": 1,
    "parental_education_level": "High School",
    "internet_quality": "Average",
    "mental_health_rating": 6,
    "extracurricular_participation": "Yes"
}])

# Предсказываем экзаменационный балл
predicted_score = model.predict(new_student)[0]
predicted_category = get_category(predicted_score)

print("\nПРОГНОЗ ДЛЯ НОВОГО СТУДЕНТА")
print(f"Прогнозируемый балл: {predicted_score:.2f}")
print(f"Прогнозируемая категория: {predicted_category}")
