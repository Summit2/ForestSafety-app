from flask import Flask, request, jsonify
from io import BytesIO
import rasterio
import numpy as np
import cv2
from onnx_model import OnnxModel
from image_preparation import prepareImage, prepareMask
from polygons_preparation import simplifyPolygons,polygonsToGeopolygons
from flask_cors import CORS  # ← импортируем CORS

# Инициализация модели
MODEL_PATH = "model_unet.onnx"
model = OnnxModel(MODEL_PATH)

app = Flask(__name__)
CORS(app)  # ← разрешаем CORS для всех маршрутов по умолчанию
'''
source ./env/bin/activate
curl -X POST -F "file1=@output1.tif" -F "file2=@output2.tif" http://127.0.0.1:5000/get_polygons/
'''

def pipeline(dataImg1 : dict, dataImg2 : dict):
    """Обрабатывает изображения и возвращает полигоны в географических координатах."""
    img1, img2 = np.transpose(dataImg1['img'], axes=(1, 2, 0)), np.transpose(dataImg2['img'], axes=(1, 2, 0))
    
    bounds = dataImg1['bounds']
    affine_transformation = dataImg1['transform']

    # Сжатие изображения до 256x256
    # img1_resized = resizeImage(np.transpose(img1, axes=(2, 0, 1)), (256,256))
    # img2_resized = resizeImage(np.transpose(img2, axes=(2, 0, 1)), (256,256))
    
    concatedImage = np.concatenate((img1,img2), axis=2)

    preparedImage = prepareImage(concatedImage)[np.newaxis, ...]  # (1, 256, 256, 16)

    assert preparedImage.shape == (1,256,256,16), f"Неправильная размерность подготовленной картинки : {preparedImage.shape}"
    
    # Предсказание модели
    mask = model.predict(preparedImage)[0][0]

    

    # Обработка маски
    preparedMask = prepareMask(mask)

    # Получение полигонов
    polygons = simplifyPolygons(preparedMask, epsilon=0.05)

    geo_polygons = polygonsToGeopolygons(polygons, 
                                       dataImg1['transform'],
                                       dataImg1['coordinatesType'])
    
    return geo_polygons




#curl -X POST -F "file1=@output_1.tif" -F "file2=@output_2.tif" http://127.0.0.1:5000/get_polygons/
@app.route("/get_polygons/", methods=["POST"])
def process_geotiff():
    """Обрабатывает два GeoTIFF файла и возвращает сегментированные полигоны."""
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "Необходимо передать два файла: file1 и file2"}), 400

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    print(f"Получен файл 1: {file1.filename}")
    print(f"Получен файл 2: {file2.filename}")

    dataImg1, dataImg2 = dict(), dict()
    # Читаем первый файл
    try:
        content1 = file1.read()
        with BytesIO(content1) as f1:
            with rasterio.open(f1) as dataset1:
                dataImg1['img'] = dataset1.read()
                dataImg1['bounds'] = dataset1.bounds
                dataImg1['coordinatesType'] = dataset1.crs
                dataImg1['transform'] = dataset1.transform
    except Exception as e:
        return jsonify({"error": f"Ошибка при чтении file1: {str(e)}"}), 400


    # print(crs)
    # Читаем второй файл
    try:
        content2 = file2.read()
        with BytesIO(content2) as f2:
            with rasterio.open(f2) as dataset2:
                dataImg2['img'] = dataset2.read()
                dataImg2['bounds'] = dataset2.bounds
                dataImg2['coordinatesType'] = dataset2.crs
                dataImg2['transform'] = dataset2.transform
    except Exception as e:
        return jsonify({"error": f"Ошибка при чтении file2: {str(e)}"}), 400


    
    

    # Проверка, что изображения имеют одинаковую размерность
    if dataImg1['img'].shape != dataImg2['img'].shape:
        
        return jsonify({"error": "Изображения должны иметь одинаковую размерность"}), 400

    if dataImg1['bounds'] != dataImg2['bounds']:
        
        return jsonify({"error": "Изображения должны быть в однаковых границах"}), 400


    # Обработка через pipeline
    polygons = pipeline(dataImg1, dataImg2)
    print(f"Найдено полигонов: {len(polygons)}")

    # Преобразование полигонов в JSON
    result = [{"points": polygon} for polygon in polygons] if polygons else []
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
