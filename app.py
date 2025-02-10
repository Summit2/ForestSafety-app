from flask import Flask, request, jsonify
from io import BytesIO
import rasterio
import numpy as np
from onnx_model import OnnxModel
from image_preparation import prepareImage, prepareMask
from polygons_preparation import simplifyPolygons

# Инициализация модели
MODEL_PATH = "model_unet.onnx"
model = OnnxModel(MODEL_PATH)

app = Flask(__name__)


def show_img(img):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.show()

def pipeline(img1, img2):
    # Подготовка картинки
    img1, img2 = np.transpose(img1, axes=(1,2,0)),np.transpose(img2, axes=(1,2,0))
    concatedImage = np.concatenate((img1,img2), axis=2)
    
    # print((concatedImage/65536).shape)


    preparedImage = prepareImage(concatedImage)[np.newaxis, ...]  # 26 channels total, 4 axis

    
    assert preparedImage.shape == (1,256,256,16), f"Неправильная размерность подготовленной картинки : {preparedImage.shape}"
    
    # Предсказание моделью
    mask = model.predict(preparedImage)[0][0]
    
    preparedMask = prepareMask(mask)

    # Преобразование маски в полигон
    polygons = simplifyPolygons(preparedMask, epsilon=0.05)

    print(len(polygons))
    return polygons


#curl -X POST -F "file1=@/home/ilya/programs/diploma/MapViewApp/backend/output_1_13.tif" -F "file2=@/home/ilya/programs/diploma/MapViewApp/backend/output_14_27.tif" http://127.0.0.1:5000/get_polygons/

@app.route("/get_polygons/", methods=["POST"])
def process_geotiff():
    # Проверка, что файлы переданы
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "Необходимо передать два файла: file1 и file2"}), 400

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    # Логирование имен файлов
    print(f"Получен файл 1: {file1.filename}")
    print(f"Получен файл 2: {file2.filename}")
    

    # Чтение первого geoTIFF файла
    try:
        content1 = file1.read()
        with BytesIO(content1) as f1:
            print(f"Размер file1: {len(content1)} байт")
            with rasterio.open(f1) as dataset1:
                img1 = dataset1.read()  # Чтение всех каналов изображения
                print(f"Успешно прочитан file1: {img1.shape}")
    except Exception as e:
        return jsonify({"error": f"Ошибка при чтении file1: {str(e)}"}), 400

    # Чтение второго geoTIFF файла
    try:
        content2 = file2.read()
        with BytesIO(content2) as f2:
            print(f"Размер file2: {len(content2)} байт")
            with rasterio.open(f2) as dataset2:
                img2 = dataset2.read()  # Чтение всех каналов изображения
                print(f"Успешно прочитан file2: {img2.shape}")
    except Exception as e:
        return jsonify({"error": f"Ошибка при чтении file2: {str(e)}"}), 400

    # Проверка, что изображения имеют одинаковую размерность
    if img1.shape != img2.shape:
        return jsonify({"error": "Изображения должны иметь одинаковую размерность"}), 400

    # Прогон через pipeline
    polygons = pipeline(img1, img2)

    # Преобразование полигонов в требуемый формат
    result = [{"points": polygon.tolist()} for polygon in polygons] if len(polygons) != 0 else []

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)