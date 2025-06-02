# Для запуска этого приложения сначала установите необходимые библиотеки:
import streamlit as st
from transformers import pipeline
import torch # Для проверки доступности GPU (хотя и не обязательно)

# Заголовок приложения
st.title('Сервис Суммаризации Текста на английском (Text Summarization)')
st.write("Использует предобученную нейросеть для краткого изложения вашего текста.")

# Загрузка предобученной модели
# Используем st.cache_resource, чтобы модель загружалась и инициализировалась только один раз
@st.cache_resource
def load_summarizer_model():
    # Можно использовать 't5-small' для меньшего размера и быстрее, но иногда менее качественно
    # Можно использовать 'sshleifer/distilbart-cnn-12-6' для лучшего качества, но требует больше памяти
    model_name = "sshleifer/distilbart-cnn-12-6"

    with st.spinner(f"Загрузка модели '{model_name}' (может занять некоторое время при первом запуске)..."):
        # Проверяем доступность GPU
        device = 0 if torch.cuda.is_available() else -1 # 0 для GPU, -1 для CPU
        summarizer = pipeline("summarization", model=model_name, device=device)
    st.success("Модель успешно загружена!")
    return summarizer

summarizer = load_summarizer_model()

st.write("---")
st.write("### Введите текст для суммаризации:")

# Поле для ввода текста пользователем
# Можно установить min_chars для минимальной длины, чтобы суммаризация имела смысл
input_text = st.text_area("Вставьте ваш текст сюда:", height=250)

# Кнопка для запуска суммаризации
if st.button('Суммаризировать'):
    if input_text:
        with st.spinner('Суммаризация текста...'):
            try:
                # Параметры для суммаризации
                # min_length: минимальная длина суммарного текста
                # max_length: максимальная длина суммарного текста
                # num_beams: количество лучей для поиска (больше - лучше качество, дольше)
                # do_sample: если True, то генерируется более разнообразный текст
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
