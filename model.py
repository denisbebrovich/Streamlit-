# Для запуска этого приложения необходимо установить библиотеки:
import streamlit as st
from transformers import pipeline
import torch 

# Заголовок приложения
st.title('Сервис Суммаризации Текста на английском (Text Summarization)')
st.write("Использует предобученную нейросеть для краткого изложения вашего текста.")

# Загрузка предобученной модели
# Используем st.cache_resource, чтобы модель загружалась и инициализировалась только один раз
@st.cache_resource
def load_summarizer_model():
    model_name = "sshleifer/distilbart-cnn-12-6"

    with st.spinner(f"Загрузка модели '{model_name}' (может занять некоторое время при первом запуске)..."):
        # Проверяем доступность GPU
        device = 0 if torch.cuda.is_available() else -1 # 0 для GPU, -1 для CPU
        summarizer = pipeline("summarization", model=model_name, device=device)
    st.success("Модель готова к использованию!")
    return summarizer

summarizer = load_summarizer_model()

st.write("---")
st.write("### Введите текст для суммаризации:")

# Поле для ввода текста пользователем
input_text = st.text_area("Вставьте ваш текст сюда:", height=250)

# Кнопка для запуска суммаризации
if st.button('Суммаризировать'):
    if input_text:
        with st.spinner('Суммаризация текста...'):
            try:
                summary_output = summarizer(
                    input_text,
                    min_length=30,  # Минимальная длина суммарного текста
                    max_length=150, # Максимальная длина суммарного текста
                    num_beams=4,    # Количество лучей для поиска (для лучшего качества)
                    do_sample=False # Не использовать сэмплирование для более детерминированного вывода
                )
                summary_text = summary_output[0]['summary_text']
                st.subheader('Результат Суммаризации:')
                st.info(summary_text) # Вывод суммарного текста
            except Exception as e:
                st.error(f"Произошла ошибка при суммаризации: {e}")
                st.warning("Пожалуйста, убедитесь, что текст достаточно длинный и понятный.")
    else:
        st.warning("Пожалуйста, введите текст для суммаризации.")
