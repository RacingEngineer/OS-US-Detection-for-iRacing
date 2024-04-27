import irsdk
from oclock import Timer
from src.rt_inference import RTDataPreprocessing, RT_OS_US_Inference, load_anfis_model, open_json

if __name__ == "__main__":
    INFERENCE_SAMPLE_TIME = 1 / 60
    # set model hyperparameters
    nummfs = 3
    numvars = 6
    # set variable to True if model was trained with VelocityX feature
    velocity_used = True
    SAVE_WEIGHTS_PATH = "data/models/mx5_vars6/state_dict_model.pt"
    NORMALIZATION_PARAMETERS_PATH = "data/models/mx5_vars6/normalization_parameters.json"
    normalization_parameters = open_json(NORMALIZATION_PARAMETERS_PATH)
    data_processing = RTDataPreprocessing(INFERENCE_SAMPLE_TIME, normalization_parameters)
    model = load_anfis_model(SAVE_WEIGHTS_PATH, mfs_nums=nummfs, varnums=numvars)
    inference = RT_OS_US_Inference(data_processing, model)
    ir = irsdk.IRSDK()
    ir.startup()
    timer = Timer(interval=INFERENCE_SAMPLE_TIME)
    while True:
        ir.freeze_var_buffer_latest()
        if velocity_used:
            indicator = inference(ir['SteeringWheelAngle'], ir['LatAccel'], ir['YawRate'], ir['VelocityX'])
        else:
            indicator = inference(ir['SteeringWheelAngle'], ir['LatAccel'], ir['YawRate'])
        print("Indicator", indicator)
        timer.checkpt()