with open('es-en/europarl-v7.es-en.en', 'r', encoding='utf-8') as file_en, \
     open('es-en/europarl-v7.es-en.es', 'r', encoding='utf-8') as file_es, \
     open("second_dataset.txt", "w", encoding='utf-8') as f_out:

    while True:
        line_en = file_en.readline()
        line_es = file_es.readline()

        # Si uno de los archivos se acaba, rompemos el bucle
        if not line_en or not line_es:
            break
        
        phrase = line_en.strip() + "\t" + line_es.strip() + "\n"
        f_out.write(phrase)
