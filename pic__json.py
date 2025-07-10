from ultralytics import YOLO
import torch
import cv2
import json
import os
from symspellpy import SymSpell, Verbosity
from gigachat import GigaChat
from dotenv import load_dotenv



def evaluate_task_gigachat(task_number, student_answer, sample_answer, api_key):
    """
    Оценивает ответ ученика через GigaChat.
    
    :param task_number: номер задания (int)
    :param student_answer: ответ ученика
    :param sample_answer: эталонный ответ
    :param api_key: ключ к GigaChat API
    :return: оценка (например, "3")
    """
    prompt = f"Ты — эксперт в области оценки работ. Твоя задача — проанализировать предоставленную работу, сравнить её с образцовым примером выполнения задания и выставить объективную оценку на основе заданных критериев(ВАЖНО!!!! в ответе необходимо указать одно число(оценку), никакие пояснения не нужны только число 0, 1, 2 или 3). Для этого следуй инструкции:\nВходные данные:\n1.Образцоое выполнение: Пример работы, написанный на максимальный балл{sample_answer}.\n2.Максимальный балл: Количество баллов, которые можно получить за идеальное выполнение задания - {3}. \n3.Работа для оценки: Текст работы, которую необходимо оценить{student_answer}.\nИнструкция:\n1.Анализ образцового выполнения: Изучи образцовое выполнение задания и выдели ключевые элементы, которые делают его идеальным (например, структура, аргументация, использование источников, стиль изложения и т.д.).\n2.Сравнение с работой для оценки: Сравни работу для оценки с образцовым выполнением. Обрати внимание на соответствие критериям оценки.\nФормат вывода:\n1.Итоговая оценка: Укажи общий балл."

    try:
        with GigaChat(credentials=api_key, ca_bundle_file="C:\\Users\\Пользователь\\Downloads\\russian_trusted_root_ca.cer") as giga:
            response = giga.chat(prompt)
            evaluation = response.choices[0].message.content.strip()

            # Проверяем, является ли результат числом
            if evaluation.isdigit() and 0 <= int(evaluation) <= 3:
                return int(evaluation)
            else:
                print(f"[WARNING] Не удалось распознать оценку для задания {task_number}: {evaluation}")
                return None
    except Exception as e:
        print(f"[ERROR] Ошибка при обращении к GigaChat: {e}")
        return None
    

def draw_text_from_json(image, json_path, yolo_boxes, threshold=0.1):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print("Ошибка открытия файла:", e)
        return {}

    words_data = []


    def recursive_find_words(data):
        if isinstance(data, dict):
            if 'word' in data and 'x' in data and 'y' in data and 'w' in data and 'h' in data:
                x = data.get('x')
                y = data.get('y')
                w = data.get('w')
                h = data.get('h')
                ttext = data.get('word', '')
                if None not in (x, y, w, h):
                    words_data.append((int(x), int(y), int(w), int(h), ttext))
            else:
                for key, value in data.items():
                    recursive_find_words(value)
        elif isinstance(data, list):
            for item in data:
                recursive_find_words(item)

    recursive_find_words(data)

    result_dict = {}

    for idx, (x1, y1, x2, y2, class_name) in enumerate(yolo_boxes):
        task_number = extract_task_number(class_name)
        if task_number is None:
            continue  

        found_texts = []
        for (tx, ty, tw, th, ttext) in words_data:
            tx2 = tx + tw
            ty2 = ty + th

            center_x_text = (tx + tx2) // 2
            center_y_text = (ty + ty2) // 2

            if (x1 - 50 <= center_x_text <= x2 + 50) and (y1 - 50 <= center_y_text <= y2 + 50):
                found_texts.append(ttext)
                image = cv2.putText(image, ttext, (x1, y1 - 10 - len(found_texts)*15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 255), 1)

        full_text = ' '.join(found_texts)
        result_dict[task_number] = full_text.strip()  # Теперь ключ — это int: 22, 23, 24
        print(f"[Задание {task_number}]: {full_text[:100]}...")

    return result_dict


def extract_task_number(class_name):
    """
    Извлекает номер задания из строки вида 'Ex_22'
    :param class_name: строка вида 'Ex_22'
    :return: int номер задания, например 22
    """
    try:
        return int(class_name.split('_')[1])
    except (IndexError, ValueError):
        print(f"[WARNING] Не удалось извлечь номер задания из: {class_name}")
        return None


def correct_spelling_in_dict(text_dict, sym_spell, max_edit_distance=2):
    """
    Принимает словарь {ключ: "текст"}, возвращает новый словарь с исправленными опечатками.
    
    :param text_dict: Входной словарь с текстами
    :param sym_spell: Экземпляр SymSpell со словарём
    :param max_edit_distance: Максимальное расстояние редактирования
    :return: Новый словарь с исправленными текстами
    """
    corrected_dict = {}

    for key, raw_text in text_dict.items():
        words = raw_text.split()
        corrected_words = []

        for word in words:
            original_word = word
            cleaned_word = preprocess_word(original_word)

            if not cleaned_word:
                corrected_words.append(original_word)
                continue

            ends_with_dash = original_word.endswith('-')
            if ends_with_dash:
                original_word = original_word[:-1]
                cleaned_word = preprocess_word(original_word)

            if not cleaned_word.isalpha():
                corrected_words.append(original_word)
                continue

            suggestions = sym_spell.lookup(cleaned_word.lower(), Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
            if suggestions:
                corrected_word = suggestions[0].term
                if original_word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                else:
                    corrected_word = corrected_word.lower()

                if ends_with_dash:
                    corrected_word += '-'
            else:
                corrected_word = original_word

            corrected_words.append(corrected_word)

        corrected_text = ' '.join(corrected_words)
        corrected_dict[key] = corrected_text

    return corrected_dict


def load_spell_checker(dictionary_path, term_index=0, count_index=1):
    """
    Загружает словарь для исправления опечаток.
    
    :param dictionary_path: Путь к словарю с частотами слов
    :param term_index: Индекс столбца со словами в словаре
    :param count_index: Индекс столбца с частотой появления
    :return: Экземпляр SymSpell с загруженным словарём
    """
    sym_spell = SymSpell()
    sym_spell.load_dictionary(dictionary_path, term_index=term_index, count_index=count_index, encoding='utf-8')
    return sym_spell


def preprocess_word(word):
    return word.strip(".,!?-—\"'()[]{}").strip()


def correct_spelling(text, sym_spell, max_edit_distance=2):
    corrected_words = []
    for word in text.split():
        original_word = word
        word = preprocess_word(word)

        if not word:
            corrected_words.append(original_word)
            continue

        # Если слово заканчивается на дефис — убираем его
        ends_with_dash = original_word.endswith('-')
        if ends_with_dash:
            original_word = original_word[:-1]

        # Пропускаем числа и смешанные слова
        if not word.isalpha():
            corrected_words.append(original_word)
            continue

        suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
        if suggestions:
            corrected_word = suggestions[0].term
            # Сохраняем регистр
            if original_word[0].isupper():
                corrected_word = corrected_word.capitalize()
            else:
                corrected_word = corrected_word.lower()

            # Восстанавливаем дефис, если был
            if ends_with_dash:
                corrected_word += '-'
        else:
            corrected_word = original_word

        corrected_words.append(corrected_word)

    return ' '.join(corrected_words)


def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")

    '''найти норм 28е задание'''
    katal = {22:'No 22\n1) Потому что противовирусный препарат может оказать влияние на количество вирусных часниц в образце. из\n2) Отрицаательный контроль не позволит выявить связь добавления препарата прочков вич на выполничество вирусных частиц. /n3) Чтобы правильно поставить отриц. контроль, без необходимо провести эксперименты без добавления препарата, а остальные условия оставить теме же.',
             23:'23 задание \n1) Нет, он не разрушает вирусные частицы так как после 120 часов их количество увеличилось или осталось тем же. Он может приостанав-ливать развитие вирусных частиц. \n2) Микроскопия \n3) РНК-вирусы в своем геноме содержат РНК. Прежде чем этот вирус попадает внутрь клетки, он с помощью обратной транскрипции синтезирует клетки ДНК вирусную и встраивает ее в геном человека При обратной транскрипции часто происходят ошибки, поэтому РНК вирусы должны постоянно мутировать поэтому у таких вирусов нет единственного лекарства \n4) У вирусов нет собственного биосиптеза белка (нет рибосом) Они используют рибосомы клетки для биосинтца вирусных белков, поэтому нет возможности воздействовать на рибосомы для остановки их размножения',
             24:'24 задание \n1) На рисунке А изображен безусловный рефлекс слюноотделения и отсутствует безразличный раздражитель - свет \n2) в продолговатом мозге \n3) инстинкт - сложная врожденная форма поведения состоящая из цепочки (нескольких) безусловных рефлексов. Безусловный рефлекс более простой. \n4) рецепторы зрительного анализатора реагируют на свет => импульс по зрительным нервам передается в кору головного мозга => образование нейроной связи между до этого безразличным раздражителем и пищевым (свет- еда) => информация доходит до продолговатого мозга в слюноотделительный отдел => двигательные нейроны => выделение слюпы',
             25:'25. 1) У жирадов достаточно длинная шея,по сравнению с человеком, соответственно, чтобы сонная артерия донесла кровь от сердца к головному мору, потребуется достаточное для этого давление. \n2) Тело жирафа массивнее, давление в каниллядах и сосудах больше => соединительная ткань должна соответствовать этим факторам иначе капиляры могут лопнуть. \n3) Капилляры в нихених конечностях не справляются с таким давлением, они начинают разбухать и это приводит к отёку. (плюс кровь концентируется именно в нижних конечностях),',
             26:'26 задание \n1) В Южной Америке обитали более древние предки сумчатый. На филогенетическом древе видно что все сумпатые виды пошли от тех, кто обитает в Южной Америке. \n2) Теория дрейдоя континентов. Изначально был один большой материк затем он разделился на 3 (Африка Южная Америка Австралия) В Южную Америку попали часть сумчатых древних предков, а в Австралию другая часть из исходной популяции. В новой среде обитания они накапливали мутации приспосабливались к условиям новой среды обитания => возникли новые виды сумчатых (В Австралии видов сумчатых стало больше поскольку для них там более благо приятные условия мало естественных врагов, хищников, много пищевых ресурсов, мало конкуренции)',
             27:'27. 1) мутантный фенотип представлен исключительно гомозиотами (aa) \n2) нормальный фенотип представлен гомороготами по домин доминантному аллелю (АА) и гетерозигатами (Aa) \n3) частота мутантного фенотипа в человеческой популяции составляет 0,0001 \n4)частота мутантного аллеля в человеческой популяции составляет 0,01 \n5) частота нормального аллеля в человеческой популяции составляет 0,99 \n6) частота нормального фенотика составляет 0,939 (AA)q^2 = 0,9801, (Aa)pq = 0,0198 \n7)Естественный отбор.',
             28:'28 задание: \n3) Да, возможно рождение ребенка в первом браке с нормальным развитием кисти и с гипертрихозом - хауан он получила от гаметы матери и кроссоверной гаметы у'}
    
    image_path = r'tests\tests_photo\1019641987_02.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.1

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_dir = os.path.dirname(image_path).replace('tests_photo', 'texts_json')
    json_path = os.path.join(json_dir, f"{base_name}__1_res.txt.webRes")

    if not os.path.exists(json_path):
        print(f"[ERROR] Соответствующий JSON-файл не найден: {json_path}")
        return

    print(f"[INFO] Используется JSON-файл: {json_path}")

    # Загрузка модели YOLO
    model = YOLO('runs\\detect\\yolov8n_custom9\\weights\\best.pt')
    model.to(device)

    results = model(image_path)

    image = cv2.imread(image_path)
    all_boxes = []  # Теперь будет содержать (x1, y1, x2, y2, class_name)

# Обработка результатов YOLO
    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()
            score = box.conf.item()
            c = int(box.cls.item())  # Приводим к int

            if score >= threshold:
                class_name = model.names[c]  # Получаем имя класса: 'Ex_22', 'Ex_23' и т.д.
                x1, y1, x2, y2 = map(int, b)
                all_boxes.append((x1, y1, x2, y2, class_name))  # добавили class_name
                print(f"[DEBUG] Найден объект: {class_name}, координаты: ({x1}, {y1}) - ({x2}, {y2})")
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][c % 3]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    extracted_texts = draw_text_from_json(image, json_path, all_boxes, threshold)

    dict_path = r'c:\Users\Пользователь\Desktop\prog\ru_full_glower.txt'  # Путь к твоему словарю с весами
    sym_spell = load_spell_checker(dict_path, term_index=0, count_index=1)

    print("\n[INFO] Исправляем опечатки в каждом тексте...")
    corrected_dict = correct_spelling_in_dict(extracted_texts, sym_spell)

    for task_num in sorted(extracted_texts.keys()):
        original = extracted_texts[task_num]
        corrected = corrected_dict[task_num]

        if original != corrected:
            print(f"\n[Задание {task_num}]")
            print("ДО:     ", original)
            print("ПОСЛЕ:  ", corrected)
        else:
            print(f"[Задание {task_num}] Ошибок не найдено.")

    # Сохранение результата
    output_file = f"output_corrected_{base_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corrected_dict, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Результат с исправленными опечатками сохранён в файл: {output_file}")

    # --- Оценка заданий через GigaChat ---
    print("\n[INFO] Оцениваем задания через GigaChat...")

    evaluations = {}

    for task_num in sorted(corrected_dict.keys()):
        student_answer = corrected_dict[task_num]
        sample_answer = katal.get(task_num)

        if not sample_answer:
            print(f"[WARNING] Нет эталона для задания {task_num}")
            continue

        print(f"\n[Задание {task_num}] Отправляем на оценку...")
        score = evaluate_task_gigachat(task_num, student_answer, sample_answer, api_key)

        if score is not None:
            evaluations[task_num] = score
            print(f"Оценка за задание {task_num}: {score}")
        else:
            print(f"Не удалось оценить задание {task_num}")

    # Сохраняем оценки в файл
    eval_output_file = f"evaluations_{base_name}.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Оценки сохранены в файл: {eval_output_file}")

    # Отображение изображения
    image = cv2.resize(image, (1000, 700))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
