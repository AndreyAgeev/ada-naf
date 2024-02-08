from PIL import Image, ImageDraw, ImageFont


if __name__ == "__main__":
    # загрузка изображений
    path = "/Users/andreyageev/PycharmProjects/NAF/res_final"
    # image_a = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_arrythmia.png')
    # image_b = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_credit.png')
    # image_c = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_diabetes.png')
    # image_d = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_eeg_eye.png')
    # image_e = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_haberman.png')
    # image_f = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_http.png')
    image_g = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_ionosphere.png')
    image_k = Image.open(f'{path}/output_num_seeds_3_num_cross_val_3_num_trees_150_count_epoch_150_contaminations_5_20230629_110543.txt_mulcross.png')
    image_l = Image.open(f'{path}/output_num_seeds_1_num_cross_val_1_num_trees_150_count_epoch_150_contaminations_5_20230702_132012.txt_seismic_bumps.png')
    # image_m = Image.open(f'{path}/output_num_seeds_1_num_cross_val_1_num_trees_150_count_epoch_150_contaminations_5_20230702_132012.txt_shuttle.png')

    # создание нового изображения
    new_image = Image.new('RGB', (2560, 1700), (255, 255, 255))

    # расположение изображений на новом изображении
    new_image.paste(image_g, (0, 10))
    new_image.paste(image_k, (600, 800))
    new_image.paste(image_l, (1270, 10))
    # new_image.paste(image_d, (1200, 800))
    # new_image.paste(image_e, (0, 1600))
    # new_image.paste(image_f, (1200, 1600))
    # new_image.paste(image_g, (0, 2400))
    # new_image.paste(image_k, (1200, 2400))

    # добавление подписей
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype('/Users/andreyageev/Downloads/arial.ttf', 50)

    # draw.text((600, 800), 'a) Arrythmia', fill=(0, 0, 0), font=font)
    # draw.text((1800, 800), 'b) Pima (Diabetes)', fill=(0, 0, 0), font=font)
    # draw.text((1200, 1600), 'c) Credit', fill=(0, 0, 0), font=font)
    # draw.text((600, 800), 'd) Eeg eye', fill=(0, 0, 0), font=font)
    # draw.text((1800, 800), 'e) Http', fill=(0, 0, 0), font=font)
    # draw.text((1200, 1600), 'f) Haberman', fill=(0, 0, 0), font=font)
    draw.text((600, 800), 'g) Ionosphere', fill=(0, 0, 0), font=font)
    draw.text((1800, 800), 'k) Sesmic bumps', fill=(0, 0, 0), font=font)
    draw.text((1200, 1600), 'l) Mulcross', fill=(0, 0, 0), font=font)
    # draw.text((1200, 1965), 'm) Shuttle', fill=(0, 0, 0), font=font)

    # сохранение нового изображения
    new_image.save('/Users/andreyageev/PycharmProjects/NAF/res_final/combined_image_3.jpg')
