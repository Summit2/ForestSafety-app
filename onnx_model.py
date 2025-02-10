import onnx
import onnxruntime

class OnnxModel():
    def __init__(self, modelName):
        self.onnxModel = onnx.load(modelName)
        onnx.checker.check_model(modelName)

        self.session = onnxruntime.InferenceSession(modelName)

        # Получение имен входных и выходных узлов модели
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name


    def predict(self, img):
        result = self.session.run([self.output_name], {self.input_name: img})
        return result